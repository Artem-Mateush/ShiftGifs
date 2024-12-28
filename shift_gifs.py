#!/usr/bin/env python3
"""Create phase-shifted video grids using python-ffmpeg."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import appeal
from ffmpeg.asyncio import FFmpeg
from tqdm import tqdm


# Constants
FRAME_RATE_PARTS = 2
DEFAULT_COLS = 3
DEFAULT_ROWS = 2
DEFAULT_SCALE_FACTOR = 1.0
DEFAULT_BITRATE_FACTOR = 1.0

class BraceLogRecord(logging.LogRecord):
    """LogRecord with support for {}-style formatting."""
    def getMessage(self) -> str:  # noqa: N802
        """Return properly formatted message with {} style support."""
        msg = str(self.msg)
        if self.args:
            msg = msg.format(*self.args) if "{" in msg else msg % self.args
        return msg

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logging(*, verbose:bool=False) -> None:
    """Configure logging based on verbosity mode."""
    logging.setLogRecordFactory(BraceLogRecord)

    logger.handlers.clear()

    if verbose:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
app = appeal.Appeal()

def _parse_progress_line(line: str) -> dict:
    """Parse a single line of FFmpeg progress output."""
    if not line:
        return {}

    # Split the line into key-value pairs
    parts = line.decode("utf-8").strip().split("=")
    if len(parts) != 2: # noqa:PLR2004
        return {}

    return {parts[0]: parts[1]}

class VideoProcessingError(Exception):
    """Raised when video processing fails."""


def is_gif(file_path: Path) -> bool:
    """Check if the file is a GIF."""
    return file_path.suffix.lower() == ".gif"


def validate_input_file(file_path: Union[str, Path]) -> Path:
    """Validate and resolve input file path."""
    try:
        path = Path(file_path).resolve(strict=True)
        if not path.is_file():
            msg = f"Not a regular file: {path}"
            raise VideoProcessingError(msg)
    except (RuntimeError, OSError) as e:
        msg = f"Invalid input path: {e}"
        raise VideoProcessingError(msg) from e
    return path


def validate_output_file(file_path: Union[str, Path]) -> Path:
    """Validate and resolve output file path."""
    try:
        path = Path(file_path).resolve()
        if not path.parent.exists():
            msg = f"Output directory does not exist: {path.parent}"
            raise VideoProcessingError(msg)
        if path.exists() and not os.access(path, os.W_OK):
            msg = f"Output file exists but is not writable: {path}"
            raise VideoProcessingError(msg)
    except (RuntimeError, OSError) as e:
        msg = f"Invalid output path: {e}"
        raise VideoProcessingError(msg) from e
    return path


def _handle_ffprobe_error(msg: str) -> None:
    """Handle FFprobe errors by raising VideoProcessingError."""
    raise VideoProcessingError(msg)


async def get_video_info(file_path: Path) -> dict:
    """Get video information using ffprobe."""
    logger.info("Getting video info for: {}", file_path)
    try:
        # Create ffprobe command
        ffprobe = (
            FFmpeg(executable="ffprobe")
            .option("v", "warning")
            .option("select_streams", "v:0")
            .option("show_entries", "stream=width,height,r_frame_rate"
                    ",bit_rate:format=duration")
            .option("of", "json")
            .input(str(file_path))
        )

        # Execute ffprobe and get output
        stdout = await ffprobe.execute()

        if not stdout:
            _handle_ffprobe_error("FFprobe returned no output")

        probe_data = json.loads(stdout)
        logger.debug("FFprobe output: {}", probe_data)

        if "streams" not in probe_data or not probe_data["streams"]:
            _handle_ffprobe_error("No video streams found")

        stream_info = probe_data["streams"][0]
        format_info = probe_data["format"]

        # Extract fps as a float
        fps_parts = stream_info.get("r_frame_rate", "").split("/")
        if len(fps_parts) != FRAME_RATE_PARTS:
            _handle_ffprobe_error(
                f"Invalid frame rate format: {stream_info.get('r_frame_rate')}",
            )

        fps = float(fps_parts[0]) / float(fps_parts[1])

        # Get bitrate, fallback to format bitrate if stream bitrate is not available
        bitrate = stream_info.get("bit_rate")
        if not bitrate:
            # If stream bitrate is not available, try format bitrate
            bitrate = format_info.get("bit_rate")

        info = {
            "duration": float(format_info["duration"]),
            "width": int(stream_info["width"]),
            "height": int(stream_info["height"]),
            "fps": fps,
            "bitrate": bitrate,
        }
        logger.info("Video info: {}", info)
    except Exception as e:
        error = VideoProcessingError(f"Failed to get video information: {e!s}")
        logger.exception(str(error))
        raise error from e
    else:
        return info


def _verify_output_dimensions(
    output_info: dict,
    expected_width: int,
    expected_height: int,
    scale_factor: float
) -> None:
    """Verify output dimensions match expected dimensions."""
    if (output_info["width"], output_info["height"]) != (
        int(expected_width * scale_factor),
        int(expected_height * scale_factor),
    ):
        msg = (
            f"Output dimensions {output_info['width']}x{output_info['height']} "
            f"don't match expected {expected_width}x{expected_height}"
        )
        _handle_ffprobe_error(msg)


def _create_filter_complex(
        cols: int,
        rows: int,
        duration: float,
        output_type: Literal["gif", "video"] = "video",
        scale_factor: float = DEFAULT_SCALE_FACTOR,
    ) -> str:
    """Create the FFmpeg filter complex string."""
    total_frames = cols * rows
    filter_complex = []

    # Split input into streams
    splits = [f"[in{i}]" for i in range(total_frames)]
    filter_complex.append(f'split={total_frames}{"".join(splits)}')

    # Create phase-shifted versions
    for i in range(total_frames):
        shift = duration - (i * duration / total_frames)
        filter_complex.extend([
            f"[in{i}]trim=start={shift},setpts=PTS-STARTPTS[p{i}a]",
            f"[in{i}]trim=duration={shift},setpts=PTS-STARTPTS[p{i}b]",
            f"[p{i}a][p{i}b]concat[v{i}]",
        ])

    # Create layout for xstack
    inputs = "".join(f"[v{i}]" for i in range(total_frames))
    layout = []
    for row in range(rows):
        for col in range(cols):
            x = "0" if col == 0 else "+".join(f"w{i}" for i in range(col))
            y = "0" if row == 0 else "+".join(f"h{i}" for i in range(row))
            layout.append(f"{x}_{y}")
    layout_str = "|".join(layout)

    # Stack the videos
    filter_complex.append(
        f"{inputs}xstack=inputs={total_frames}:layout={layout_str}[stacked]",
    )

    # Apply scaling if needed
    if scale_factor != DEFAULT_SCALE_FACTOR:
        filter_complex.append(
            f"[stacked]scale=iw*{scale_factor}:ih*{scale_factor}[out]",
        )
    else:
        filter_complex.append("[stacked]copy[out]")

    # For GIFs, add optimization filters
    if output_type == "gif":
        filter_complex.extend([
            "[out]fps=15,split[x1][x2]",
            "[x1]palettegen=max_colors=128:stats_mode=single[p]",
            "[x2][p]paletteuse=dither=floyd_steinberg",
        ])

    return ";".join(filter_complex)

async def create_phase_grid(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
    *,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    bitrate_factor: float = DEFAULT_BITRATE_FACTOR,
) -> None:
    """Create a grid of phase-shifted videos."""
    try:
        input_path = validate_input_file(input_file)
        output_path = validate_output_file(output_file)
        is_output_gif = is_gif(output_path)

        logger.info("Processing video: {} -> {}", input_path, output_path)
        logger.info("Grid dimensions: {}x{}", cols, rows)

        # Get video information
        info = await get_video_info(input_path)
        duration = info["duration"]

        filter_complex_str = _create_filter_complex(
            cols,
            rows,
            duration,
            output_type="gif" if is_output_gif else "video",
            scale_factor=scale_factor,
        )

        # Initialize progress bar
        pbar = tqdm(
            total=100,
            desc="Processing video",
            bar_format="{l_bar}{bar}| {n_fmt}%",
            unit="%",
        )

        # Build FFmpeg command
        ffmpeg = (
            FFmpeg()
            .option("y")
            .option("v", "warning")
            .option("progress", "pipe:1")
            .option("stats_period", "0.1")
            .input(str(input_path))
        )

        last_progress = 0

        # Set output options based on format
        if is_output_gif:
            ffmpeg = (
                ffmpeg
                .output(
                    str(output_path),
                    filter_complex=filter_complex_str,
                    f="gif",
                    vsync="2",
                )
            )
        else:
            # Calculate new bitrate for the grid
            input_bitrate = info.get("bitrate")
            new_bitrate = int(int(input_bitrate) * (cols * rows) * bitrate_factor)
            logger.info("Scaling input bitrate {} to {} for grid",
                input_bitrate, new_bitrate)

            ffmpeg = (
                ffmpeg
                .output(
                    str(output_path),
                    filter_complex=filter_complex_str,
                    map="[out]",
                    c="libx264",
                    preset="veryslow",
                    crf="28",
                    pix_fmt="yuv420p",
                    b="{}".format(new_bitrate),
                    maxrate="{}".format(new_bitrate * 1.5),
                    bufsize="{}".format(new_bitrate * 2),
                    profile="baseline",
                    level="3.0",
                    tune="film",
                )
            )


        @ffmpeg.on("progress")
        def on_progress(progress: Any) -> None: # noqa: ANN401
            nonlocal last_progress
            try:
                current_time = progress.time.total_seconds()
                current_progress = min(100, int((current_time / duration) * 100))

                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
            except (AttributeError, TypeError, ValueError):
                logger.exception("Progress update failed")

        try:
            await ffmpeg.execute()

        finally:
            pbar.close()

        # Verify output dimensions
        output_info = await get_video_info(output_path)
        expected_width = info["width"] * cols
        expected_height = info["height"] * rows

        _verify_output_dimensions(output_info, expected_width, expected_height, scale_factor=scale_factor)
        logger.info("Video processing completed successfully")

    except Exception as e:
        error = VideoProcessingError(f"Video processing failed: {e!s}")
        logger.exception(str(error))
        raise error from e



def process_video(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
    *,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    bitrate_factor: float = DEFAULT_BITRATE_FACTOR,
) -> None:
    """Wrapper function to run the async create_phase_grid."""
    asyncio.run(create_phase_grid(
        input_file,
        output_file,
        cols, rows,
        scale_factor=scale_factor,
        bitrate_factor=bitrate_factor,
    ))


def _validate_geometry(cols: int, rows: int) -> None:
    """Validate the geometry values are positive."""
    if cols <= 0 or rows <= 0:
        msg = "Geometry values must be positive"
        raise ValueError(msg)

FactorType = Annotated[float, appeal.validate_range(0.001, 1.0, type=float)]
@app.global_command()
def shift_gifs(
    input_file: str,
    output_file: Annotated[Optional[str], str] = None,
    *,
    geometry: str = f"{DEFAULT_COLS}x{DEFAULT_ROWS}",
    scale: FactorType = DEFAULT_SCALE_FACTOR,
    bitrate_factor: FactorType = DEFAULT_BITRATE_FACTOR,
    verbose: bool = False,
) -> Optional[int]:
    """Create a grid of phase-shifted videos.

    Args:
        input_file: Path to input video file
        output_file: Path to output video file.
          If not provided, will use input path with '_shifted' suffix
        geometry: Grid dimensions in format 'COLSxROWS',
          e.g. '3x2' for 3 columns and 2 rows
        scale: Scale factor for output size (e.g., 0.5 for half size)
        bitrate_factor: How much reduce quality (default 0.5 for 50%)
        verbose: Verbose logging output
    """
    if output_file is None:
        input_path = Path(input_file)
        stem = input_path.stem
        output_file = str(input_path.with_stem(f"{stem}_shifted"))

    setup_logging(verbose=verbose)
    try:
        logger.info("Starting grid creation: {}", geometry)
        cols, rows = map(int, geometry.split("x"))
        _validate_geometry(cols, rows)
        process_video(input_file, output_file, cols, rows,
          scale_factor=scale, bitrate_factor=bitrate_factor)
        logger.info("Processing completed")
    except Exception:
        logger.exception("Processing failed")
        return 1
    else:
        return None

if __name__ == "__main__":
    app.main()
