# TIFF Utilities

Compact helpers for inspecting and converting OME-TIFF and whole-slide images. The python script provides a command-line interface for common slide microscopy workflows such as generating RGB previews, converting images into tiled pyramidal TIFFs, and summarising image metadata. Tested on macOS and Linux.

## Features
- Convert arbitrary images (OME-TIFF, TIFF, common raster formats, OpenSlide-readable slides) into tiled and optionally pyramidal TIFFs with compression.
- Render quick RGB previews from large multi-dimensional OME-TIFF assets.
- Inspect series, axes, and pyramid levels without fully loading the image.

## Prerequisites
- Python 3.9+ (with the standard `venv` module available)
- GNU `make`

## CLI Installation
Use the provided `Makefile` to set up the CLI wrapper in one step:
```bash
make install
```
This creates `.venv`, installs the dependencies listed in `requirements.txt`, and symlinks the script as `~/.local/bin/pytiffutil`. Include optional extras (e.g. OpenSlide support) by appending them to the `EXTRA` variable:
```bash
make EXTRA="openslide-python" install
```
To provision the virtual environment without creating the symlink run `make deps`. Remove the CLI later with `make uninstall`. Ensure `~/.local/bin` is on your `PATH` so the `pytiffutil` command is globally available.

## Command-Line Usage
The script exposes three subcommands. Replace `python tiffutils.py` with the path from your clone—or use `pytiffutil` if you created the symlink above.

### `convert`
Convert any supported image into a tiled TIFF (optionally pyramidal for OpenSlide for visualization in tools such as QuPath).
```bash
python tiffutils.py convert slide.svs slide.tif \
  --openslide 
```
- `--compression` chooses the TIFF compression scheme (`zlib`, `lzw`, `deflate`, or `none`).
- `--tile` accepts a single edge length (`1024`) or `WxH` (`1024x1024`); `none` disables tiling.
- `--bigtiff` controls BigTIFF output (`auto|yes|no`); `auto` toggles based on size.
- `--no-check-pil` skips verifying the output with Pillow.
- `--openslide`, `--openslide-levels`, `--openslide-min-dim` create an OpenSlide-compatible pyramid.
- `--series`, `--level`, `--z`, `--t`, `--c/--channels` pick subsets from multi-dimensional OME data.

### `preview`
Create a downscaled RGB PNG for quick inspection.
```bash
python tiffutils.py preview slide.ome.tif preview.png --max-size 2048
```
- `--max-size` caps the longer edge (default 2048 px).
- The same selector flags (`--series`, `--level`, `--z`, `--t`, `--c/--channels`) apply.

### `describe`
Summarise OME-TIFF metadata.
```bash
python tiffutils.py describe slide.ome.tif
```
prints a human-readable summary.

## Credits
100% of the code was conjured by Codex; I just gave instructions, poked it to see if it worked, and tried not to trip over the power cord.
