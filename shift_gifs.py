#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from typing import Optional

import appeal


app = appeal.Appeal()

def get_duration(file_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return float(result.stdout.strip())

def get_fps(file_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    num, den = map(int, result.stdout.strip().split("/"))
    return num/den

def create_layout_string(cols, rows):
    layout = []
    for row in range(rows):
        for col in range(cols):
            # Calculate cumulative positions
            if col == 0:
                x = "0"
            else:
                # Sum up all previous widths: w0+w1+...+w(n-1)
                x = "+".join(f"w{i}" for i in range(col))

            if row == 0:
                y = "0"
            else:
                # Sum up all previous heights: h0+h1+...+h(n-1)
                y = "+".join(f"h{i}" for i in range(row))

            layout.append(f"{x}_{y}")
    return "|".join(layout)

def get_dimensions(file_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        file_path,
    ]
    dimensions = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    width, height = map(int, dimensions.split("x"))
    return width, height

def create_phase_grid(input_file, output_file, cols=3, rows=2) -> None:
    if not Path(input_file).exists():
        sys.exit(1)

    duration = get_duration(input_file)
    get_fps(input_file)
    total_frames = cols * rows
    in_width, in_height = get_dimensions(input_file)
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
        f"{inputs}xstack=inputs={total_frames}:layout={layout}:shortest=1[vs];" +
        f"[vs]scale={out_width}:{out_height},format=yuv420p[v]",
    )

    # Build complete ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",
        "-i", input_file,
        "-filter_complex", "".join(filter_complex),
        "-map", "[v]",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "medium",
        output_file,
    ]


    subprocess.run(cmd, check=False)

    # Verify output dimensions
    out_actual_width, out_actual_height = get_dimensions(output_file)

    if (out_actual_width, out_actual_height) != (out_width, out_height):
        pass


@app.global_command()
def phase_grid(input_file: str, output_file: str, geometry: str = "3x2") -> Optional[int]:
    """Create a grid of phase-shifted videos."""
    try:
        cols, rows = map(int, geometry.split("x"))
        if cols <= 0 or rows <= 0:
            msg = "Geometry values must be positive"
            raise ValueError(msg)
    except ValueError:
        return 1

    create_phase_grid(input_file, output_file, cols, rows)
    return None

if __name__ == "__main__":
    app.main()
