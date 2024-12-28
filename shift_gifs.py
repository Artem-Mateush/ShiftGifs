#!/usr/bin/env python3
"""Create phase-shifted video grids using python-ffmpeg."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional, Union

import appeal
from ffmpeg import FFmpeg


# Constants
FRAME_RATE_PARTS = 2
DEFAULT_COLS = 3
DEFAULT_ROWS = 2


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


class VideoProcessingError(Exception):
    """Raised when video processing fails."""


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


def get_video_info(file_path: Path) -> dict:
    """Get video information using ffprobe."""
    logger.info("Getting video info for: {}", file_path)
    try:
        # Create ffprobe command
        ffprobe = (
            FFmpeg(executable="ffprobe")
            .option("v", "error")
            .option("select_streams", "v:0")
            .option("show_entries", "stream=width,height,r_frame_rate:format=duration")
            .option("of", "json")
            .input(str(file_path))
        )

        # Execute ffprobe and get output
        result = ffprobe.execute()

        # Parse the JSON output
        stdout = result[0] if isinstance(result, tuple) else result

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

        info = {
            "duration": float(format_info["duration"]),
            "width": int(stream_info["width"]),
            "height": int(stream_info["height"]),
            "fps": fps,
        }
        logger.info("Video info: {}", info)
    except Exception as e:
        error = VideoProcessingError(f"Failed to get video information: {e!s}")
        logger.exception(str(error))
        raise error from e
    else:
        return info


def _verify_output_dimensions(
    output_info: dict, expected_width: int, expected_height: int,
) -> None:
    """Verify output dimensions match expected dimensions."""
    if (output_info["width"], output_info["height"]) != (
        expected_width,
        expected_height,
    ):
        msg = (
            f"Output dimensions {output_info['width']}x{output_info['height']} "
            f"don't match expected {expected_width}x{expected_height}"
        )
        _handle_ffprobe_error(msg)


def create_phase_grid(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
) -> None:
    """Create a grid of phase-shifted videos."""
    try:
        input_path = validate_input_file(input_file)
        output_path = validate_output_file(output_file)

        logger.info("Processing video: {} -> {}", input_path, output_path)
        logger.info("Grid dimensions: {}x{}", cols, rows)

        # Get video information
        info = get_video_info(input_path)
        duration = info["duration"]

        # Build filter complex
        total_frames = cols * rows
        filter_complex = []

        # Split input into streams
        splits = [f"[in{i}]" for i in range(total_frames)]
        filter_complex.append(f'split={total_frames}{"".join(splits)}')

        # Create phase-shifted versions
        for i in range(total_frames):
            shift = duration - (i * duration / total_frames)
            filter_complex.extend(
                [
                    f"[in{i}]trim=start={shift},setpts=PTS-STARTPTS[p{i}a]",
                    f"[in{i}]trim=duration={shift},setpts=PTS-STARTPTS[p{i}b]",
                    f"[p{i}a][p{i}b]concat[v{i}]",
                ],
            )

        # Create layout for xstack
        inputs = "".join(f"[v{i}]" for i in range(total_frames))
        layout = []
        for row in range(rows):
            for col in range(cols):
                x = "0" if col == 0 else "+".join(f"w{i}" for i in range(col))
                y = "0" if row == 0 else "+".join(f"h{i}" for i in range(row))
                layout.append(f"{x}_{y}")
        layout_str = "|".join(layout)

        # Add xstack filter
        filter_complex.append(
            f"{inputs}xstack=inputs={total_frames}:layout={layout_str}:shortest=1[vs]",
        )
        filter_complex.append("[vs]format=yuv420p[v]")

        # Join all filters
        filter_complex_str = ";".join(filter_complex)
        logger.debug("Filter complex: {}", filter_complex_str)

        # Create FFmpeg command
        ffmpeg = (
            FFmpeg()
            .option("y")
            .option("stream_loop", "-1")
            .input(str(input_path))
            .output(
                str(output_path),
                {"filter_complex": filter_complex_str},
                map="[v]",
                t=str(duration),
                **{"c:v": "libx264", "preset": "medium"},
            )
        )

        # Execute FFmpeg command
        result = ffmpeg.execute()

        # Handle potential stderr output
        if isinstance(result, tuple) and len(result) > 1:
            stderr = result[1]
            if stderr:
                logger.warning("FFmpeg stderr: {}", stderr)

        # Verify output dimensions
        output_info = get_video_info(output_path)
        expected_width = info["width"] * cols
        expected_height = info["height"] * rows

        _verify_output_dimensions(output_info, expected_width, expected_height)
        logger.info("Video processing completed successfully")

    except Exception as e:
        error = VideoProcessingError(f"Video processing failed: {e!s}")
        logger.exception(str(error))
        raise error from e


def _validate_geometry(cols: int, rows: int) -> None:
    """Validate the geometry values are positive."""
    if cols <= 0 or rows <= 0:
        msg = "Geometry values must be positive"
        raise ValueError(msg)


@app.global_command()
def phase_grid(
    input_file: str,
    output_file: Annotated[Optional[str], str] = None,
    *,
    geometry: str = f"{DEFAULT_COLS}x{DEFAULT_ROWS}",
    verbose: bool = False,
) -> Optional[int]:
    """Create a grid of phase-shifted videos.

    Args:
        input_file: Path to input video file
        output_file: Path to output video file.
          If not provided, will use input path with '_shifted' suffix
        geometry: Grid dimensions in format 'COLSxROWS',
          e.g. '3x2' for 3 columns and 2 rows
        verbose: Verbose logging output
    """
    if output_file is None:
        # Get input path without extension
        input_path = Path(input_file)
        stem = input_path.stem
        # Create output path with _shifted suffix and same extension
        output_file = str(input_path.with_stem(f"{stem}_shifted"))

    setup_logging(verbose=verbose)
    try:
        logger.info("Starting grid creation: {}", geometry)
        cols, rows = map(int, geometry.split("x"))
        _validate_geometry(cols, rows)
        create_phase_grid(input_file, output_file, cols, rows)
        logger.info("Processing completed")
    except Exception:
        logger.exception("Processing failed")
        return 1
    else:
        return None

if __name__ == "__main__":
    app.main()
