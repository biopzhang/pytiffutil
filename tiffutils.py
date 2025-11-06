#!/usr/bin/env python3
# Requirements: numpy, Pillow, imageio[v3], tifffile, openslide-python (optional)

from __future__ import annotations
import os
import logging
import json
import warnings
from contextlib import contextmanager
from typing import Optional, Sequence, Dict, Any, List, Tuple, Iterator
import numpy as np
from PIL import Image
import imageio.v3 as iio
import tifffile as tiff  # type: ignore

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover - optional
    openslide = None

Image.MAX_IMAGE_PIXELS = None  # disable PIL max pixel limit
logger = logging.getLogger(__name__)

try:  # Pillow < 10 compatibility
    _LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def _is_incompatible_keyframe_error(err: Exception) -> bool:
    """Return True if the tifffile error is due to incompatible keyframes."""
    return isinstance(err, RuntimeError) and "incompatible keyframe" in str(err).lower()


class _TifffileShapeWarningFilter(logging.Filter):
    """Suppress specific noisy tifffile warnings."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        return self._text not in record.getMessage()


@contextmanager
def _suppress_tifffile_shape_warning() -> Iterator[None]:
    """Temporarily silence the 'shaped series shape does not match page shape' warning."""
    message = "shaped series shape does not match page shape"
    tf_logger = logging.getLogger("tifffile")
    log_filter = _TifffileShapeWarningFilter(message)
    tf_logger.addFilter(log_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*shaped series shape does not match page shape.*",
                category=UserWarning,
                module="tifffile",
            )
            yield
    finally:
        tf_logger.removeFilter(log_filter)


def _read_openslide_image(path: str, *, level: Optional[int] = None) -> np.ndarray:
    """Load a whole-slide image via openslide at the requested level."""
    if openslide is None:  # pragma: no cover - optional dependency
        raise ImportError("openslide-python is required to read this image format.")
    slide = openslide.OpenSlide(path)  # type: ignore[attr-defined]
    try:
        lvl = int(level or 0)
        lvl = max(0, min(lvl, slide.level_count - 1))
        width, height = slide.level_dimensions[lvl]
        region = slide.read_region((0, 0), lvl, (width, height))
        arr = np.asarray(region)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            alpha = arr[..., 3]
            if np.all(alpha == 255):
                arr = arr[..., :3]
            else:
                arr = arr[..., :3]
        return arr
    finally:
        slide.close()


def _prepare_openslide_rgb(arr: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 RGB array suitable for OpenSlide pyramids."""
    data = np.asarray(arr)
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    elif data.ndim == 3:
        if data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        elif data.shape[-1] >= 3:
            data = data[..., :3]
        else:
            raise ValueError("Unsupported channel configuration for OpenSlide export.")
    else:
        raise ValueError("OpenSlide export requires 2D or 3D image arrays.")

    data = _to_uint8(data)
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    if data.shape[-1] == 4:
        data = data[..., :3]
    if data.shape[-1] != 3:
        raise ValueError("Unable to coerce image to 3-channel RGB for OpenSlide export.")
    return np.ascontiguousarray(data.astype(np.uint8, copy=False))


def _downsample_by_two(arr: np.ndarray) -> np.ndarray:
    """Half-resolution downsample using Pillow LANCZOS."""
    h, w = arr.shape[0], arr.shape[1]
    if h <= 1 or w <= 1:
        return arr
    new_size = (max(1, w // 2), max(1, h // 2))
    if new_size == (w, h):
        return arr
    img = Image.fromarray(arr)
    return np.asarray(img.resize(new_size, _LANCZOS))


def _build_pyramid_levels(
    base: np.ndarray,
    *,
    max_levels: int = 8,
    min_dim: int = 512,
) -> List[np.ndarray]:
    """Construct a list of downsampled levels for an OpenSlide pyramid."""
    levels = [base]
    current = base
    for _ in range(max_levels - 1):
        if min(current.shape[0], current.shape[1]) <= min_dim:
            break
        nxt = _downsample_by_two(current)
        if nxt.shape[:2] == current.shape[:2]:
            break
        levels.append(np.ascontiguousarray(nxt))
        current = nxt
    return levels


def _write_openslide_tiff(
    dst_path: str,
    base_image: np.ndarray,
    *,
    tile: Tuple[int, int],
    compression: Optional[str],
    bigtiff: bool,
    metadata: Optional[Dict[str, Any]] = None,
    max_levels: int = 8,
    min_level_dim: int = 512,
) -> None:
    """Write a pyramidal, tiled TIFF compatible with OpenSlide."""
    rgb = np.asarray(base_image)
    if not (rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8):
        rgb = _prepare_openslide_rgb(rgb)
    pyramid = _build_pyramid_levels(rgb, max_levels=max_levels, min_dim=min_level_dim)

    write_opts = dict(
        tile=tuple(tile),
        compression=compression,
        photometric="rgb",
        planarconfig="contig",
    )

    with tiff.TiffWriter(dst_path, bigtiff=bigtiff) as tif:
        tif.write(
            pyramid[0],
            subfiletype=0,
            metadata=metadata,
            **write_opts,
        )
        for level in pyramid[1:]:
            tif.write(
                level,
                subfiletype=1,
                metadata=None,
                **write_opts,
            )


def _to_uint8(
    arr: np.ndarray,
    *,
    in_range: str | tuple[float, float] = "image",  # "image" | "dtype" | (lo, hi)
    ignore_zeros: bool = False,                     # ignore 0s when finding lo/hi
    per_channel: bool = True,                      # scale each channel separately if HxWxC
    percentiles: tuple[float, float] | None = (1, 99) # e.g., (1, 99) for robust scaling
) -> np.ndarray:
    """
    Convert image array to uint8 with configurable input range handling.

    - Integers or floats supported.
    - If percentiles is set, lo/hi come from those percentiles (on nonzero pixels if ignore_zeros=True).
    - If per_channel=True and arr.ndim==3, scaling is applied channel-wise.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a  # already uint8

    # Helper to get lo/hi for a 2D (or flat) view
    def _get_lo_hi(x: np.ndarray) -> tuple[float, float]:
        if isinstance(in_range, tuple):
            lo, hi = float(in_range[0]), float(in_range[1])
        elif in_range == "dtype":
            if np.issubdtype(x.dtype, np.integer):
                info = np.iinfo(x.dtype)
                lo, hi = float(info.min), float(info.max)
            else:
                # float dtype="dtype" is ambiguous; fall back to image stats
                lo = float(np.nanmin(x))
                hi = float(np.nanmax(x))
        elif in_range == "image":
            if ignore_zeros:
                nz = x[x != 0]
                if nz.size == 0:
                    return 0.0, 1.0
                base = nz
            else:
                base = x
            if percentiles is not None:
                p_lo, p_hi = percentiles
                lo = float(np.nanpercentile(base, p_lo))
                hi = float(np.nanpercentile(base, p_hi))
            else:
                lo = float(np.nanmin(base))
                hi = float(np.nanmax(base))
        else:
            raise ValueError("in_range must be 'image', 'dtype', or (lo, hi)")
        # Handle degenerate cases
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
        return lo, hi

    # Fast path for 2D or 1-channel
    if a.ndim < 3 or (a.ndim == 3 and a.shape[-1] == 1):
        lo, hi = _get_lo_hi(a)
        out = (np.clip(a.astype(np.float32), lo, hi) - lo) / (hi - lo)
        return (out * 255.0 + 0.5).astype(np.uint8)

    # HxWxC path
    if a.ndim == 3 and per_channel:
        out = np.empty(a.shape, dtype=np.uint8)
        for c in range(a.shape[-1]):
            lo, hi = _get_lo_hi(a[..., c])
            ch = (np.clip(a[..., c].astype(np.float32), lo, hi) - lo) / (hi - lo)
            out[..., c] = (ch * 255.0 + 0.5).astype(np.uint8)
        return out
    else:
        # single scaling for all channels combined
        lo, hi = _get_lo_hi(a)
        out = (np.clip(a.astype(np.float32), lo, hi) - lo) / (hi - lo)
        return (out * 255.0 + 0.5).astype(np.uint8)


def _select_channels(arr: np.ndarray, channels: Optional[Sequence[int]]) -> np.ndarray:
    """Put channels last and keep requested channels (default: first 3)."""
    if arr.ndim == 2:
        return arr
    # move channels to last if they look like a channel axis
    if arr.shape[-1] in (1, 3, 4):
        ch_last = arr
    elif arr.shape[0] in (1, 3, 4):
        ch_last = np.moveaxis(arr, 0, -1)
    else:
        ch_last = arr  # unknown layout; leave as-is

    if ch_last.ndim == 3:
        C = ch_last.shape[-1]
        if channels is None:
            if C >= 3:
                ch_last = ch_last[..., :3]
            elif C == 2:
                z = np.zeros_like(ch_last[..., :1])
                ch_last = np.concatenate([ch_last, z], axis=-1)
            elif C == 1:
                ch_last = ch_last[..., 0]
        else:
            idx = [c for c in channels if 0 <= c < C]
            if len(idx) == 1:
                ch_last = ch_last[..., idx[0]]
            elif len(idx) >= 2:
                ch_last = ch_last[..., idx]
    return ch_last


def _rgb_from_multichannel(arr: np.ndarray, channels: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return an 8-bit RGB array from a multichannel image.

    - Accepts HxW, HxW x C or C x H x W.
    - If channels is None: use first 3 if available; if 2 channels, pad with zeros; if 1, replicate to 3.
    - Scales to uint8.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        g = _to_uint8(arr)
        return np.stack([g, g, g], axis=-1)

    # Move channels to last if needed
    if arr.shape[-1] in (1, 2, 3, 4):
        ch_last = arr
    elif arr.shape[0] in (1, 2, 3, 4):
        ch_last = np.moveaxis(arr, 0, -1)
    else:
        ch_last = arr

    if ch_last.ndim != 3:
        ch_last = np.squeeze(ch_last)
        if ch_last.ndim == 2:
            g = _to_uint8(ch_last)
            return np.stack([g, g, g], axis=-1)

    C = ch_last.shape[-1]
    if channels is None:
        if C >= 3:
            sel = ch_last[..., :3]
        elif C == 2:
            z = np.zeros_like(ch_last[..., :1])
            sel = np.concatenate([ch_last, z], axis=-1)
        else:
            g = _to_uint8(ch_last[..., 0])
            return np.stack([g, g, g], axis=-1)
    else:
        idx = [c for c in channels if 0 <= c < C]
        if len(idx) == 1:
            g = _to_uint8(ch_last[..., idx[0]])
            return np.stack([g, g, g], axis=-1)
        elif len(idx) >= 3:
            sel = ch_last[..., idx[:3]]
        elif len(idx) == 2:
            z = np.zeros_like(ch_last[..., :1])
            sel = np.concatenate([ch_last[..., idx], z], axis=-1)
        else:
            g = _to_uint8(ch_last[..., 0])
            return np.stack([g, g, g], axis=-1)

    return _to_uint8(sel)


def preview_ome(
    path: str,
    *,
    series: Optional[int] = None,
    level: Optional[int] = None,
    z: Optional[int] = None,
    t: Optional[int] = None,
    c: Optional[Sequence[int] | int] = None,
    channels: Optional[Sequence[int]] = None,
    max_size: int = 2048,
) -> Image.Image:
    """Generate a downscaled RGB preview for an OME-TIFF/TIFF.

    - Selects series/level/z/t/c as provided (defaults to 0 when present)
    - If multichannel and no channels provided, uses first 3 channels
    - Scales to fit within max_size on the longer side
    - Returns a PIL RGB image suitable for quick inspection
    """
    sel_c = channels if channels is not None else c

    info = describe_ome(path)
    is_tif = info.get("is_tiff", False)
    if not is_tif:
        im = read_image(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.thumbnail((max_size, max_size))
        return im

    if tiff is None:  # pragma: no cover
        raise ImportError("tifffile is required for preview_ome. Install via 'pip install tifffile'.")

    with _suppress_tifffile_shape_warning():
        with tiff.TiffFile(path) as tf:  # type: ignore[attr-defined]
            try:
                s_idx = int(series or 0)
                s_idx = max(0, min(s_idx, len(tf.series) - 1))
                s = tf.series[s_idx]

                lvl_idx = int(level) if level is not None else (len(getattr(s, "levels", [])) - 1 or 0)
                lvl_idx = max(0, min(lvl_idx, len(getattr(s, "levels", [])) - 1)) if getattr(s, "levels", None) else 0
                s_use = s.levels[lvl_idx] if getattr(s, "levels", None) and lvl_idx > 0 else s

                arr = s_use.asarray()
                axes = getattr(s_use, "axes", getattr(s, "axes", ""))
            except Exception as err:
                if not _is_incompatible_keyframe_error(err):
                    raise
                logger.debug("tifffile series detection failed (%s); falling back to first page", err)
                if len(tf.pages) == 0:
                    raise RuntimeError("No TIFF pages found in file") from err
                page_idx = int(z) if z is not None else 0
                page_idx = max(0, min(page_idx, len(tf.pages) - 1))
                page = tf.pages[page_idx]
                arr = page.asarray()
                axes = getattr(page, "axes", "")
                if not axes:
                    axes = "YXS" if arr.ndim == 3 else "YX"

    axes = str(axes)
    indexer: List[Any] = []
    channel_axis: Optional[int] = None
    for dim_i, ax in enumerate(axes):
        if ax in ("Y", "X"):
            indexer.append(slice(None))
        elif ax in ("C", "S"):
            channel_axis = dim_i
            if sel_c is None:
                indexer.append(slice(None))
            elif isinstance(sel_c, int):
                indexer.append(int(sel_c))
            else:
                indexer.append(slice(None))
        elif ax == "Z":
            indexer.append(int(z) if z is not None else 0)
        elif ax == "T":
            indexer.append(int(t) if t is not None else 0)
        else:
            indexer.append(0)

    arr = arr[tuple(indexer)]

    if channel_axis is not None and not isinstance(sel_c, int) and sel_c is not None:
        arr = np.moveaxis(arr, channel_axis if arr.ndim > channel_axis else -1, -1)
        idx = [int(cidx) for cidx in sel_c if 0 <= int(cidx) < arr.shape[-1]]
        if len(idx) == 1:
            arr = np.stack([arr[..., idx[0]]] * 3, axis=-1)
        elif len(idx) >= 2:
            if len(idx) == 2:
                zpad = np.zeros_like(arr[..., :1])
                arr = np.concatenate([arr[..., idx], zpad], axis=-1)
            else:
                arr = arr[..., idx[:3]]

    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    rgb = _rgb_from_multichannel(arr, None if isinstance(sel_c, int) else sel_c)
    #print("Preview RGB shape:", rgb.shape)
    #print(f"{rgb.max()} {rgb.min()} {rgb.dtype}")
    im = Image.fromarray(rgb, mode="RGB")
    im.thumbnail((max_size, max_size))
    return im


def read_image(
    path: str,
    *,
    series: Optional[int] = None,
    level: Optional[int] = None,
    t: Optional[int] = None,
    z: Optional[int] = None,
    c: Optional[Sequence[int] | int] = None,
    channels: Optional[Sequence[int]] = None,  # deprecated alias of c
    preserve_dtype: bool = False,
) -> Image.Image:
    """
    Read an image. Uses imageio.v3 (tifffile) for TIFF/OME-TIFF, else PIL.
    Always returns a PIL.Image.Image.

    Args:
        path: File path/URI.
        series: For multi-series OME-TIFFs, which series to read (0-based).
        z: For 3D stacks, which z-slice to take (0-based).
        channels: Which channel indices to keep; default picks first 3 if available.
        preserve_dtype: Keep 16-bit grayscale as 16-bit PIL ('I;16') if True.

    Returns:
        PIL.Image.Image
    """
    ext = os.path.splitext(path)[1].lower()

    # Prefer tifffile for TIFF/OME-TIFF to precisely select axes/levels
    if ext in {".tif", ".tiff"} and tiff is not None:
        try:
            with _suppress_tifffile_shape_warning():
                with tiff.TiffFile(path) as tf:  # type: ignore[attr-defined]
                    try:
                        series_index = int(series or 0)
                        series_index = max(0, min(series_index, len(tf.series) - 1))
                        s = tf.series[series_index]

                        # Handle pyramidal levels if requested
                        if level is not None and getattr(s, "levels", None):
                            lvl_index = int(level)
                            lvl_index = max(0, min(lvl_index, len(s.levels) - 1))  # type: ignore[attr-defined]
                            s_lvl = s.levels[lvl_index]  # type: ignore[index]
                            arr = s_lvl.asarray()
                            axes = getattr(s_lvl, "axes", getattr(s, "axes", ""))
                        else:
                            arr = s.asarray()
                            axes = getattr(s, "axes", "")
                    except Exception as err:
                        if not _is_incompatible_keyframe_error(err):
                            raise
                        logger.debug("tifffile series detection failed (%s); falling back to first page", err)
                        if len(tf.pages) == 0:
                            raise RuntimeError("No TIFF pages found in file") from err
                        page_idx = int(z) if z is not None else 0
                        page_idx = max(0, min(page_idx, len(tf.pages) - 1))
                        page = tf.pages[page_idx]
                        arr = page.asarray()
                        axes = getattr(page, "axes", "")
                        if not axes:
                            axes = "YXS" if arr.ndim == 3 else "YX"

            # Build slicing for provided axes selections
            axes = str(axes)
            sel_c = channels if channels is not None else c

            # Map each axis to index/slice
            indexer: List[Any] = []
            channel_axis: Optional[int] = None
            for dim_i, ax in enumerate(axes):
                if ax in ("Y", "X"):
                    indexer.append(slice(None))
                elif ax in ("C", "S"):
                    channel_axis = dim_i
                    if sel_c is None:
                        indexer.append(slice(None))
                    elif isinstance(sel_c, int):
                        indexer.append(int(sel_c))
                    else:
                        # list/seq of channels -> slice None for now; we'll apply after moveaxis
                        indexer.append(slice(None))
                elif ax == "Z":
                    indexer.append(int(z) if z is not None else 0)
                elif ax == "T":
                    indexer.append(int(t) if t is not None else 0)
                else:
                    # Unknown/other dims (S, R, etc.): take first
                    indexer.append(0)

            # Apply basic indexing
            arr = arr[tuple(indexer)]

            # If we have a channel sequence selection, apply it now after moving channel last
            if channel_axis is not None and not isinstance(sel_c, int) and sel_c is not None:
                arr = np.moveaxis(arr, channel_axis if arr.ndim > channel_axis else -1, -1)
                idx = [int(cidx) for cidx in sel_c if 0 <= int(cidx) < arr.shape[-1]]
                if len(idx) == 1:
                    arr = arr[..., idx[0]]
                elif len(idx) >= 2:
                    arr = arr[..., idx]

            # Normalize shape to 2D or 3D (H,W[,C])
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.moveaxis(arr, 0, -1)

            # Convert numpy -> PIL
            if arr.ndim == 2:
                if preserve_dtype and arr.dtype == np.uint16:
                    return Image.fromarray(arr, mode="I;16")
                return Image.fromarray(_to_uint8(arr), mode="L")
            elif arr.ndim == 3:
                arr8 = _to_uint8(arr)
                if arr8.shape[-1] == 3:
                    return Image.fromarray(arr8, mode="RGB")
                if arr8.shape[-1] == 4:
                    return Image.fromarray(arr8, mode="RGBA")
                return Image.fromarray(arr8[..., :3], mode="RGB")
            else:
                arr = np.squeeze(arr)
                return Image.fromarray(_to_uint8(arr))
        except Exception as e:  # pragma: no cover - fallback
            logger.debug(f"tifffile read failed, falling back to imageio/PIL: {e}")

    # Fallback: imageio for TIFF or PIL for others
    try:
        arr: Optional[np.ndarray] = None
        sel_c = channels if channels is not None else c if isinstance(c, Sequence) else None

        if ext not in {".tif", ".tiff"} and openslide is not None:
            try:
                arr = _read_openslide_image(path, level=level)
            except Exception as e_os:
                logger.debug("openslide read failed for %s: %s", path, e_os)

        if arr is None:
            arr = iio.imread(path)

        arr = _select_channels(arr, sel_c)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            if preserve_dtype and arr.dtype == np.uint16:
                return Image.fromarray(arr, mode="I;16")
            return Image.fromarray(_to_uint8(arr), mode="L")
        elif arr.ndim == 3:
            arr8 = _to_uint8(arr)
            if arr8.shape[-1] == 3:
                return Image.fromarray(arr8, mode="RGB")
            if arr8.shape[-1] == 4:
                return Image.fromarray(arr8, mode="RGBA")
            return Image.fromarray(arr8[..., :3], mode="RGB")
    except Exception:
        pass

    # Non-TIFF or TIFF fallback: use PIL, then normalize modes for consistency
    im = Image.open(path)
    if im.mode in ("I;16", "I", "F") and not preserve_dtype:
        im = im.convert("L")
    if im.mode not in ("RGB", "RGBA", "L", "I;16"):
        im = im.convert("RGB")
    return im


def describe_ome(path: str) -> Dict[str, Any]:
    """Return a lightweight description of an OME-TIFF/TIFF file structure.

    Provides per-series information with axes strings and shapes so callers can
    pick the appropriate series/level/Z/T/C indices (e.g., from spaceranger metadata).

    Returns a dict with keys: 'is_tiff', 'is_ome' (if available), and 'series'.
    Each series item includes: index, shape, axes, dtype, n_levels.
    """
    info: Dict[str, Any] = {"is_tiff": False, "is_ome": None, "series": []}
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".tif", ".tiff"} or tiff is None:
        return info
    try:
        with _suppress_tifffile_shape_warning():
            with tiff.TiffFile(path) as tf:  # type: ignore[attr-defined]
                info["is_tiff"] = True
                info["is_ome"] = getattr(tf, "is_ome", None)

                try:
                    series_iter = list(tf.series)
                except Exception as err:
                    if not _is_incompatible_keyframe_error(err):
                        raise
                    logger.debug("describe_ome: tifffile series detection failed (%s); using page metadata", err)
                    if len(tf.pages) == 0:
                        raise RuntimeError("No TIFF pages found in file") from err
                    page = tf.pages[0]
                    axes = getattr(page, "axes", "")
                    if not axes:
                        ndim = getattr(page, "ndim", None)
                        axes = "YXS" if ndim == 3 else "YX"
                    shape = tuple(getattr(page, "shape", ()))
                    dtype = str(getattr(page, "dtype", ""))
                    if (not shape or not dtype) and hasattr(page, "asarray"):
                        try:
                            arr = page.asarray()
                            if not shape:
                                shape = tuple(arr.shape)
                            if not dtype:
                                dtype = str(arr.dtype)
                        except Exception as arr_err:
                            logger.debug("describe_ome fallback array load failed: %s", arr_err)
                    info["series"].append(
                        {
                            "index": 0,
                            "shape": shape,
                            "axes": axes,
                            "dtype": dtype,
                            "n_levels": 1,
                            "source": "page_fallback",
                        }
                    )
                else:
                    for idx, s in enumerate(series_iter):
                        entry = {
                            "index": idx,
                            "shape": tuple(getattr(s, "shape", ())),
                            "axes": getattr(s, "axes", ""),
                            "dtype": str(getattr(s, "dtype", "")),
                            "n_levels": len(getattr(s, "levels", []) or []),
                        }
                        info["series"].append(entry)
    except Exception as e:  # pragma: no cover
        logger.debug(f"describe_ome failed: {e}")

    if info["is_ome"] is None:
        print("not ome")

    return info

def convert_to_tif(
    src_path: str,
    dst_path: str,
    *,
    compression: str = "zlib",
    tile: Optional[Tuple[int, int] | int] = 1024,
    bigtiff: Optional[bool] = None,
    metadata: Optional[Dict[str, Any]] = None,
    check_with_pil: bool = True,
    openslide_pyramid: bool = False,
    openslide_levels: Optional[int] = None,
    openslide_min_dim: int = 512,
    # OME selectors (used when reading TIFF/OME-TIFF)
    series: Optional[int] = None,
    level: Optional[int] = None,
    z: Optional[int] = None,
    t: Optional[int] = None,
    c: Optional[Sequence[int] | int] = None,
    channels: Optional[Sequence[int]] = None,
) -> str:
    """Convert any image readable by tifffile or OpenSlide to a tiled compressed TIFF.

    - Automatically enables BigTIFF if output exceeds 4GB when bigtiff=None
    - Defaults to zlib compression and 1024x1024 tiles for large slides
    - With openslide_pyramid=True, creates a pyramidal TIFF layout readable by openslide-python

    Returns the destination path.
    """
    if tiff is None:  # pragma: no cover
        raise ImportError("tifffile is required for convert_to_tif. Install via 'pip install tifffile'.")

    # Read with tifffile (for TIFF) to preserve precision and allow OME selections
    ext = os.path.splitext(src_path)[1].lower()
    sel_c = channels if channels is not None else c
    if ext in {".tif", ".tiff"}:
        with _suppress_tifffile_shape_warning():
            with tiff.TiffFile(src_path) as tf:  # type: ignore[attr-defined]
                series_index = int(series or 0)
                series_index = max(0, min(series_index, len(tf.series) - 1))
                s = tf.series[series_index]

                # Handle pyramidal levels if requested
                if level is not None and getattr(s, "levels", None):
                    lvl_index = int(level)
                    lvl_index = max(0, min(lvl_index, len(s.levels) - 1))  # type: ignore[attr-defined]
                    s_lvl = s.levels[lvl_index]  # type: ignore[index]
                    arr = s_lvl.asarray()
                    axes = getattr(s_lvl, "axes", getattr(s, "axes", ""))
                else:
                    arr = s.asarray()
                    axes = getattr(s, "axes", "")

        # Build slicing for provided axes selections
        axes = str(axes)
        indexer: List[Any] = []
        channel_axis: Optional[int] = None
        for dim_i, ax in enumerate(axes):
            if ax in ("Y", "X"):
                indexer.append(slice(None))
            elif ax in ("C", "S"):
                channel_axis = dim_i
                if sel_c is None:
                    indexer.append(slice(None))
                elif isinstance(sel_c, int):
                    indexer.append(int(sel_c))
                else:
                    indexer.append(slice(None))
            elif ax == "Z":
                indexer.append(int(z) if z is not None else 0)
            elif ax == "T":
                indexer.append(int(t) if t is not None else 0)
            else:
                indexer.append(0)

        arr = arr[tuple(indexer)]

        # If channel sequence selection, apply after moving channel last
        if channel_axis is not None and not isinstance(sel_c, int) and sel_c is not None:
            arr = np.moveaxis(arr, channel_axis if arr.ndim > channel_axis else -1, -1)
            idx = [int(cidx) for cidx in sel_c if 0 <= int(cidx) < arr.shape[-1]]
            if len(idx) == 1:
                arr = arr[..., idx[0]]
            elif len(idx) >= 2:
                arr = arr[..., idx]

        # Normalize shape to 2D or 3D (H,W[,C])
        arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        im_arr = arr
    else:
        arr: Optional[np.ndarray] = None
        if openslide is not None:
            try:
                arr = _read_openslide_image(src_path, level=level)
            except Exception as e_os:
                logger.debug("openslide read failed for %s: %s", src_path, e_os)

        if arr is None:
            arr = iio.imread(src_path)

        if arr.ndim > 3:
            arr = np.squeeze(arr)
        # Best-effort channel selection if provided
        if arr.ndim == 3 and sel_c is not None:
            C = arr.shape[-1]
            if isinstance(sel_c, int) and 0 <= sel_c < C:
                arr = arr[..., sel_c]
            elif isinstance(sel_c, (list, tuple)):
                idx = [int(cidx) for cidx in sel_c if 0 <= int(cidx) < C]
                if len(idx) == 1:
                    arr = arr[..., idx[0]]
                elif len(idx) >= 2:
                    arr = arr[..., idx]
        im_arr = arr

    im_arr = np.asarray(im_arr)

    if isinstance(tile, int):
        tile_arg: Optional[Tuple[int, int]] = (tile, tile)
    else:
        tile_arg = tile

    compression_arg = None if compression in (None, "none") else compression

    prepared_base: Optional[np.ndarray] = None
    if openslide_pyramid:
        prepared_base = _prepare_openslide_rgb(im_arr)
        byte_count = prepared_base.nbytes
    else:
        byte_count = im_arr.nbytes

    if bigtiff is None:
        bigtiff_flag = bool(byte_count > 2**32 - 1)
    else:
        bigtiff_flag = bool(bigtiff)

    if openslide_pyramid:
        tile_tuple = tile_arg if tile_arg is not None else (512, 512)
        tile_tuple = tuple(int(v) for v in tile_tuple)
        if openslide_levels is not None:
            max_levels = max(1, int(openslide_levels))
        else:
            max_levels = 8
        min_dim = max(1, int(openslide_min_dim))
        _write_openslide_tiff(
            dst_path,
            prepared_base if prepared_base is not None else im_arr,
            tile=tile_tuple,
            compression=compression_arg,
            bigtiff=bigtiff_flag,
            metadata=metadata,
            max_levels=max_levels,
            min_level_dim=min_dim,
        )

        if check_with_pil:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(dst_path) as im:
                im.load()
        return dst_path

    # Write standard TIFF
    tiff.imwrite(
        dst_path,
        im_arr,
        bigtiff=bigtiff_flag,
        compression=compression_arg,
        tile=tile_arg,
        metadata=metadata,
    )

    if check_with_pil:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(dst_path) as im:
            im.load()

    return dst_path


def _parse_tile_arg(s: str) -> Optional[Tuple[int, int] | int]:
    """Parse tile spec like '1024' or '1024x1024'; 'none' disables tiling."""
    s = s.strip().lower()
    if s in {"none", "no", "false", "0"}:
        return None
    if "x" in s:
        a, b = s.split("x", 1)
        return (int(a), int(b))
    return int(s)


def _parse_bigtiff_arg(s: str) -> Optional[bool]:
    """Parse bigtiff flag: 'auto' -> None, 'yes'/'true' -> True, 'no'/'false' -> False."""
    s = s.strip().lower()
    if s in {"auto", "default"}:
        return None
    if s in {"yes", "true", "1"}:
        return True
    if s in {"no", "false", "0"}:
        return False
    raise ValueError("bigtiff must be one of: auto|yes|no")


def main() -> None:  # CLI for conversion and inspection
    import argparse

    parser = argparse.ArgumentParser(description="Utilities for OME-TIFF/TIFF: convert, preview, describe.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # convert subcommand
    p_conv = sub.add_parser("convert", help="Convert image to tiled/compressed TIFF (supports OpenSlide formats)")
    p_conv.add_argument("src", help="Input image path")
    p_conv.add_argument("dst", help="Output TIFF path")
    p_conv.add_argument("--compression", default="zlib", choices=["zlib", "lzw", "deflate", "none"], help="TIFF compression")
    p_conv.add_argument("--tile", default="1024", help="Tile size: e.g. '1024' or '1024x1024', or 'none' to disable")
    p_conv.add_argument("--bigtiff", default="auto", help="auto|yes|no")
    p_conv.add_argument("--no-check-pil", action="store_true", help="Skip verifying output can be read by PIL")
    p_conv.add_argument("--openslide", action="store_true", help="Write a pyramidal TIFF readable by openslide-python")
    p_conv.add_argument("--openslide-levels", type=int, default=None, help="Maximum pyramid levels (default auto)")
    p_conv.add_argument("--openslide-min-dim", type=int, default=512, help="Stop pyramid when dimensions drop below this")
    # OME selectors
    p_conv.add_argument("--series", type=int, default=None, help="OME series index (0-based)")
    p_conv.add_argument("--level", type=int, default=None, help="Pyramidal level index (0-based)")
    p_conv.add_argument("--z", type=int, default=None, help="Z index (0-based)")
    p_conv.add_argument("--t", type=int, default=None, help="T index (0-based)")
    p_conv.add_argument("--c", type=str, default=None, help="Channel index or list, e.g. '0' or '0,1,2'")
    p_conv.add_argument("--channels", type=str, default=None, help="Alias of --c")

    # preview subcommand
    p_prev = sub.add_parser("preview", help="Render a downscaled RGB preview to a PNG")
    p_prev.add_argument("src", help="Input image path")
    p_prev.add_argument("out_png", help="Output preview PNG path")
    p_prev.add_argument("--max-size", type=int, default=2048, help="Max preview size on long edge")
    p_prev.add_argument("--series", type=int, default=None, help="OME series index (0-based)")
    p_prev.add_argument("--level", type=int, default=None, help="Pyramidal level index (0-based)")
    p_prev.add_argument("--z", type=int, default=None, help="Z index (0-based)")
    p_prev.add_argument("--t", type=int, default=None, help="T index (0-based)")
    p_prev.add_argument("--c", type=str, default=None, help="Channel index or list, e.g. '0' or '0,1,2'")
    p_prev.add_argument("--channels", type=str, default=None, help="Alias of --c")

    # describe subcommand
    p_desc = sub.add_parser("describe", help="Describe OME-TIFF series/axes/levels")
    p_desc.add_argument("src", help="Input image path")
    p_desc.add_argument("--json", action="store_true", help="Output JSON to stdout")

    args = parser.parse_args()

    # Shared parsing for channel list
    def _parse_c_arg(a: Any) -> Optional[Sequence[int] | int]:
        c_arg = a.c if getattr(a, 'c', None) is not None else getattr(a, 'channels', None)
        if c_arg is not None:
            c_list = [int(s) for s in str(c_arg).split(',') if s.strip() != '']
            return c_list[0] if len(c_list) == 1 else c_list
        return None

    if args.cmd == 'preview':
        c_sel = _parse_c_arg(args)
        im = preview_ome(
            args.src,
            series=args.series,
            level=args.level,
            z=args.z,
            t=args.t,
            c=c_sel,
        )
        im.save(args.out_png)
        return

    if args.cmd == 'convert':
        tile_arg = _parse_tile_arg(args.tile)
        bigtiff_arg = _parse_bigtiff_arg(args.bigtiff)
        compression = None if args.compression == "none" else args.compression
        c_sel = _parse_c_arg(args)
        convert_to_tif(
            args.src,
            args.dst,
            compression=compression or "none",
            tile=tile_arg,
            bigtiff=bigtiff_arg,
            metadata=None,
            check_with_pil=not args.no_check_pil,
            openslide_pyramid=args.openslide,
            openslide_levels=args.openslide_levels,
            openslide_min_dim=args.openslide_min_dim,
            series=args.series,
            level=args.level,
            z=args.z,
            t=args.t,
            c=c_sel,
        )
        return

    if args.cmd == 'describe':
        info = describe_ome(args.src)
        if getattr(args, 'json', False):
            print(json.dumps(info, indent=2))
        else:
            # Human-readable output
            print(f"is_tiff: {info.get('is_tiff')}  is_ome: {info.get('is_ome')}")
            series_list = info.get('series', [])
            for s in series_list:
                idx = s.get('index')
                axes = s.get('axes')
                shape = s.get('shape')
                dtype = s.get('dtype')
                n_levels = s.get('n_levels')
                print(f"series {idx}: axes={axes} shape={shape} dtype={dtype} levels={n_levels}")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
