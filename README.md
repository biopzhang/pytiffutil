# TIFF Utilities

Compact helpers for inspecting and converting OME-TIFF and whole-slide images. The script provides both a Python API and a command-line interface for common slide microscopy workflows such as generating RGB previews, converting images into tiled pyramidal TIFFs, and summarising file structure.

## Features
- Convert arbitrary images (OME-TIFF, TIFF, common raster formats, OpenSlide-readable slides) into tiled and optionally pyramidal TIFFs with compression.
- Render quick RGB previews from large multi-dimensional OME-TIFF assets.
- Inspect series, axes, and pyramid levels without fully loading the image.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy Pillow imageio tifffile openslide-python  # openslide-python optional
```

## Command-Line Usage
The script exposes three subcommands. Replace `python tiffutils.py` with the path from your clone.

### `convert`
Convert any supported image into a tiled TIFF (optionally pyramidal for OpenSlide).
```bash
python tiffutils.py convert slide.svs slide.tif \
  --compression zlib \
  --tile 1024 \
  --openslide \
  --openslide-levels 6 \
  --c 0,1,2
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
python tiffutils.py preview slide.ome.tif preview.png --max-size 2048 --series 0 --c 0,1,2
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
100% of the code was conjured by Codex; I just gave instructions, poked it to see if it worked, and tried not to trip over the power cord.***
