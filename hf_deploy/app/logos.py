"""
Embedded logo images as base64 data URIs. No external network needed.
Loads actual institutional logos from app/logos/*.svg (MIT, ETH Zurich).
"""
import base64
from pathlib import Path

_LOGO_DIR = Path(__file__).resolve().parent / "logos"


def _svg_file_to_data_uri(filename: str) -> str:
    """Load SVG from file and return as data URI."""
    path = _LOGO_DIR / filename
    if path.exists():
        svg = path.read_text(encoding="utf-8")
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
    return ""


def mit_logo_data_uri() -> str:
    """Data URI for MIT logo (SVG from Wikimedia Commons)."""
    uri = _svg_file_to_data_uri("mit.svg")
    if uri:
        return uri
    # Fallback: minimal SVG if file missing
    return _fallback_svg("MIT", "#A31F34")


def eth_logo_data_uri() -> str:
    """Data URI for ETH Zurich logo (SVG from Wikimedia Commons)."""
    uri = _svg_file_to_data_uri("eth.svg")
    if uri:
        return uri
    return _fallback_svg("eth", "#000000")


def _fallback_svg(text: str, fill: str) -> str:
    """Minimal fallback if logo files are missing."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40" width="100" height="40">
  <text x="50" y="28" font-family="sans-serif" font-size="24" font-weight="bold" fill="{fill}" text-anchor="middle">{text}</text>
</svg>'''
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"
