"""
Display LaTeX paper figures (PDFs) in the Marimo app.
Figures are converted to high-quality PNG and displayed as images.
"""
import base64
from io import BytesIO
from pathlib import Path

import marimo as mo

from app.paths import PAPER_FIGS, LATEX_FIGS

# DPI for PDF→PNG conversion (higher = sharper, larger file)
PNG_DPI = 200

# Cache folder for converted PNGs (avoids re-converting on every run)
PNG_CACHE = LATEX_FIGS / ".png_cache"


def _pdf_to_png_bytes(path: Path, dpi: int = PNG_DPI) -> bytes | None:
    """Convert first page of PDF to PNG bytes. Returns None on failure."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        if doc.page_count == 0:
            doc.close()
            return None
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        png_bytes = pix.tobytes(output="png")
        doc.close()
        return png_bytes
    except Exception:
        return None


def _get_cached_png(path: Path) -> Path | None:
    """Return path to cached PNG if it exists and is newer than the PDF."""
    PNG_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = PNG_CACHE / f"{path.stem}.png"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return cache_path
    return None


def _convert_and_cache_pdf(path: Path, dpi: int = PNG_DPI) -> bytes | None:
    """Convert PDF to PNG, optionally cache to disk, return PNG bytes."""
    PNG_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = PNG_CACHE / f"{path.stem}.png"
    png_bytes = _pdf_to_png_bytes(path, dpi=dpi)
    if png_bytes is not None:
        try:
            cache_path.write_bytes(png_bytes)
        except OSError:
            pass  # cache write failed, still return bytes
    return png_bytes


def pdf_if_exists(key: str, width: str = "100%", height: str = "1000px"):
    """Return marimo component displaying the figure as high-quality PNG if file exists, else None."""
    # Embedded figures (HF deploy - no binary files)
    try:
        from app.embedded_figs import EMBEDDED_FIGS
        if key in EMBEDDED_FIGS:
            b64 = EMBEDDED_FIGS[key]
            html = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:{width};width:100%;height:auto;border:1px solid #ddd;border-radius:4px;" '
                f'alt="{key}" />'
            )
            return mo.Html(html)
    except ImportError:
        pass
    path = PAPER_FIGS.get(key)
    if path is None:
        return None
    # Prefer pre-generated PNG (for deploy without PDFs)
    png_path = path.with_suffix(".png")
    if png_path.exists():
        try:
            png_bytes = png_path.read_bytes()
            b64 = base64.b64encode(png_bytes).decode("ascii")
            html = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:{width};width:100%;height:auto;border:1px solid #ddd;border-radius:4px;" '
                f'alt="{png_path.name}" />'
            )
            return mo.Html(html)
        except Exception:
            pass
    if not path.exists():
        return None
    try:
        # Prefer cached PNG if up to date
        cached = _get_cached_png(path)
        if cached is not None:
            png_bytes = cached.read_bytes()
        else:
            png_bytes = _convert_and_cache_pdf(path)
        if png_bytes is None:
            return None
        b64 = base64.b64encode(png_bytes).decode("ascii")
        html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:{width};width:100%;height:auto;border:1px solid #ddd;border-radius:4px;" '
            f'alt="{path.name}" />'
        )
        return mo.Html(html)
    except Exception:
        return None


def available_fig_keys() -> list[str]:
    """Return list of figure keys whose PDF or PNG files exist."""
    return [k for k, p in PAPER_FIGS.items() if p and (p.exists() or p.with_suffix(".png").exists())]
