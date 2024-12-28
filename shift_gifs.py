#!/usr/bin/env python3
"""Create phase-shifted video grids using ffmpeg."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union

import appeal


app = appeal.Appeal()


class VideoProcessingError(Exception):
    """Raised when video processing fails."""


def validate_input_file(file_path: Union[str, Path]) -> Path:
    """Validate and resolve input file path.

    Args:
        file_path: Path to the input file

    Returns:
        Resolved Path object

    Raises:
        VideoProcessingError: If the path is invalid or file doesn't exist
    """
    try:
        path = Path(file_path).resolve(strict=True)
        if not path.is_file():
            msg = f"Not a regular file: {path}"
            raise VideoProcessingError(msg)
    except (RuntimeError, OSError) as e:
        msg = f"Invalid input path: {e}"
        raise VideoProcessingError(msg) from e
    else:
        return path

def validate_output_file(file_path: Union[str, Path]) -> Path:
    """Validate and resolve output file path.

    Args:
        file_path: Path where the output will be saved

    Returns:
        Resolved Path object

    Raises:
        VideoProcessingError: If the path is invalid or directory doesn't exist
    """
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
    else:
        return path

def run_ffmpeg_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command safely.

    Args:
        cmd: Command and arguments to run

    Returns:
        CompletedProcess instance with command output

    Raises:
        VideoProcessingError: If the command fails
    """
    # Validate that the first argument is either ffmpeg or ffprobe
    if not cmd or cmd[0] not in ("ffmpeg", "ffprobe"):
        msg = "Invalid command. Only ffmpeg and ffprobe are allowed."
        raise VideoProcessingError(msg)

    try:
        return subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )
    except subprocess.CalledProcessError as e:
        msg = f"Command failed with exit code {e.returncode}: {e.stderr}"
        raise VideoProcessingError(msg) from e


def get_duration(file_path: Path) -> float:
    """Get the duration of a video file in seconds.

    Args:
        file_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        VideoProcessingError: If duration cannot be determined
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = run_ffmpeg_command(cmd)
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        msg = "Could not determine video duration"
        raise VideoProcessingError(msg) from e


def get_fps(file_path: Path) -> float:
    """Get the frames per second of a video file.

    Args:
        file_path: Path to the video file

    Returns:
        Frames per second

    Raises:
        VideoProcessingError: If FPS cannot be determined
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = run_ffmpeg_command(cmd)
    try:
        num, den = map(int, result.stdout.strip().split("/"))
        return num / den
    except (ValueError, ZeroDivisionError) as e:
        msg = "Could not determine video FPS"
        raise VideoProcessingError(msg) from e


def create_layout_string(cols: int, rows: int) -> str:
    """Create a layout string for ffmpeg xstack filter.

    Args:
        cols: Number of columns in the grid
        rows: Number of rows in the grid

    Returns:
        A string representing the layout for ffmpeg xstack filter
    """
    layout = []
    for row in range(rows):
        for col in range(cols):
            # Calculate cumulative positions
            x = "0" if col == 0 else "+".join(f"w{i}" for i in range(col))
            y = "0" if row == 0 else "+".join(f"h{i}" for i in range(row))
            layout.append(f"{x}_{y}")
    return "|".join(layout)


def get_dimensions(file_path: Path) -> tuple[int, int]:
    """Get the width and height of a video file.

    Args:
        file_path: Path to the video file

    Returns:
        A tuple containing (width, height) of the video

    Raises:
        VideoProcessingError: If dimensions cannot be determined
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(file_path),
    ]
    result = run_ffmpeg_command(cmd)
    try:
        width, height = map(int, result.stdout.strip().split("x"))
    except ValueError as e:
        msg = "Could not determine video dimensions"
        raise VideoProcessingError(msg) from e
    else:
        return width, height


def create_phase_grid(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    cols: int = 3,
    rows: int = 2,
) -> None:
    """Create a grid of phase-shifted videos.

    Args:
        input_file: Path to the input video file
        output_file: Path where the output video will be saved
        cols: Number of columns in the grid
        rows: Number of rows in the grid

    Raises:
        VideoProcessingError: If processing fails
    """
    input_path = validate_input_file(input_file)
    output_path = validate_output_file(output_file)

    duration = get_duration(input_path)
    get_fps(input_path)
    total_frames = cols * rows
    in_width, in_height = get_dimensions(input_path)
    out_width = in_width * cols
    out_height = in_height * rows

    layout = create_layout_string(cols, rows)

    # Build filter complex
    splits = [f"[in{i}]" for i in range(total_frames)]
    filter_complex = [
        f'split={total_frames}{"".join(splits)};',
    ]

    # Create phase-shifted versions
    for i in range(total_frames):
        # Reverse the shift calculation: start from duration and go down
        shift = duration - (i * duration / total_frames)
        filter_complex.extend([
            f"[in{i}]trim=start={shift},setpts=PTS-STARTPTS[p{i}a];",
            f"[in{i}]trim=duration={shift},setpts=PTS-STARTPTS[p{i}b];",
            f"[p{i}a][p{i}b]concat[v{i}];",
        ])

    # Add final stack with explicit dimensions
    inputs = "".join(f"[v{i}]" for i in range(total_frames))
    filter_complex.append(
        f"{inputs}xstack=inputs={total_frames}:layout={layout}:shortest=1[vs];"
        f"[vs]scale={out_width}:{out_height},format=yuv420p[v]",
    )

    # Build complete ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",
        "-i", str(input_path),
        "-filter_complex", "".join(filter_complex),
        "-map", "[v]",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "medium",
        str(output_path),
    ]

    run_ffmpeg_command(cmd)

    # Verify output dimensions
    out_actual_width, out_actual_height = get_dimensions(output_path)
    if (out_actual_width, out_actual_height) != (out_width, out_height):
        msg = (
            f"Output dimensions {out_actual_width}x{out_actual_height} "
            f"don't match expected {out_width}x{out_height}"
        )
        raise VideoProcessingError(msg)


def _validate_geometry(cols: int, rows: int) -> None:
    """Validate the geometry values are positive.

    Args:
        cols: Number of columns
        rows: Number of rows

    Raises:
        ValueError: If either cols or rows is not positive
    """
    if cols <= 0 or rows <= 0:
        msg = "Geometry values must be positive"
        raise ValueError(msg)


@app.global_command()
def phase_grid(input_file: str, output_file: str,
    geometry: str = "3x2") -> Optional[int]:
    """Create a grid of phase-shifted videos.

    Args:
        input_file: Path to the input video file
        output_file: Path where the output video will be saved
        geometry: Grid dimensions in the format "colsxrows" (e.g., "3x2")

    Returns:
        None on success, 1 on error
    """
    try:
        cols, rows = map(int, geometry.split("x"))
        _validate_geometry(cols, rows)
        create_phase_grid(input_file, output_file, cols, rows)
    except (ValueError, VideoProcessingError):
        return 1
    else:
        return None


if __name__ == "__main__":
    app.main()
