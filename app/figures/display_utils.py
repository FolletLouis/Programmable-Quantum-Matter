"""
Utilities for displaying matplotlib figures in Marimo.
Static PNG rendering avoids mo.mpl.interactive websocket issues when cells re-run.
"""
import base64
from io import BytesIO

import marimo as mo


def fig_to_html(fig, dpi: int = 100, max_width: str = "100%") -> mo.Html:
    """Convert matplotlib figure to static HTML image. More stable than mo.mpl.interactive."""
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return mo.Html(
        f'<img src="data:image/png;base64,{b64}" style="max-width:{max_width};height:auto;" alt="plot" />'
    )
