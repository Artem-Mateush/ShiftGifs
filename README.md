# ShiftGifs

Create phase-shifted video grids using python-ffmpeg.

## Description

This tool takes a video file and creates a grid of phase-shifted versions of that video. Each cell in the grid starts at a different point in the video's timeline, creating an interesting visual effect when played together.

## Requirements

- Python 3.8+
- FFmpeg installed on your system

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shiftgifs.git
   cd shiftgifs
   ```

2. Install dependencies using uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install appeal python-ffmpeg
   ```

## Usage

Basic usage:
```bash
./shift_gifs.py input.mp4
```
This will create `input_shifted.mp4` with a default 3x2 grid.

### Options

- Specify custom output file:
  ```bash
  ./shift_gifs.py input.mp4 output.mp4
  ```

- Change grid dimensions (e.g., 4x3 grid):
  ```bash
  ./shift_gifs.py input.mp4 --geometry 4x3
  ```

- Enable verbose logging:
  ```bash
  ./shift_gifs.py input.mp4 --verbose
  ```

### Full Usage

```bash
./shift_gifs.py [-h] [--geometry GEOMETRY] [--verbose] input_file [output_file]

Arguments:
  input_file            Path to input video file
  output_file          Path to output video file (optional, defaults to input_shifted.mp4)

Options:
  --geometry GEOMETRY   Grid dimensions in format 'COLSxROWS' (default: 3x2)
  --verbose            Enable verbose logging output
  -h, --help           Show this help message and exit
```

## Example

Converting a video of a seagull:
```bash
./shift_gifs.py seagull.mp4 --geometry 3x3 --verbose
```
This creates a 3x3 grid of phase-shifted seagull videos, where each cell starts at a different point in the timeline.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.```