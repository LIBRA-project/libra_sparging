"""Single matplotlib style for every figure of the thesis and the paper.

`morethemes`' "minimal" theme sets the major tick length to zero while leaving the minor ticks
visible, which reads as if the axis had lost its ticks. Everything here is applied *after* the
theme so that choice is overridden in one place rather than per figure.

Usage::

    from sparging.plot_style import apply, categorical_axis
    apply()
    ...
    categorical_axis(ax, "x")   # for bar/violin axes, where minor ticks are meaningless
"""

import os

import matplotlib.pyplot as plt

SERIF = ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"]

# a colour-blind safe set (Tol bright), used consistently across figures
COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
}

RC = {
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    # ticks: inward, and majors actually visible (the theme zeroes them)
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "legend.frameon": False,
    "hatch.linewidth": 0.3,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,      # embed TrueType, so the text stays selectable and editable
    "ps.fonttype": 42,
}


def apply(theme: str = "minimal", **overrides) -> None:
    """Set the theme, then the thesis style on top of it.

    Also pins SOURCE_DATE_EPOCH, which matplotlib uses as the PDF creation date. Without it
    every regenerated figure differs from the last one in its metadata alone, so a rerun shows
    up as a change to every figure in git even when nothing was redrawn.
    """
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    try:
        import morethemes as mt

        mt.set_theme(theme)
    except ImportError:
        pass
    plt.rcParams.update(RC)
    if overrides:
        plt.rcParams.update(overrides)


def categorical_axis(ax, which: str = "x") -> None:
    """Drop minor ticks on a categorical axis, where they carry no meaning.

    Bar charts, violins and any axis whose positions are group indices rather than a
    continuous coordinate.
    """
    for axis in which:
        ax.tick_params(axis=axis, which="minor", **{
            {"x": "bottom", "y": "left"}[axis]: False
        })
