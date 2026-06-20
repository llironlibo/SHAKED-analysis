"""
Figure 1 -- CONSORT-style participant flow diagram.

Draws the enrollment / allocation / analysis flow: presentation to the ED, wing
allocation (intervention vs control), exclusions, the analyzed cohorts, and the
per-protocol split into SHAKED users vs non-users, with ITT and PP zones.

Reproduces: Figure 1 (CONSORT flow + engagement hierarchy).

Inputs : none (counts are fixed from the enrollment flow)
Outputs: results/figures/Figure1_CONSORT.png and .pdf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle


# =============================================================================
# Style
# =============================================================================

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


STYLE: Dict[str, Any] = {
    # Typography
    "fs_title": 12,           # Main box labels
    "fs_box": 11,             # Secondary box labels
    "fs_box_small": 10,       # Excluded boxes
    "fs_sample_size": 10,     # Sample sizes
    "fs_zone_label": 12,      # Zone labels
    "fw_zone_label": "bold",
    # Stroke and dashes
    "stroke": 1.0,            # Base stroke for exclusion arrows
    "stroke_major": 1.5,      # Major flow arrows
    "stroke_heavy": 1.5,      # PP boundary
    "dash_pp": (4, 3),
    "arrowstyle": "->",
    "arrow_shrink": 0.5,
    "box_rounding": 0.6,
    # Colors (desaturated)
    "c_text": "#000000",
    "c_border": "#333333",
    "c_grey": "#E8E8E8",
    "c_zone": "#F0F0F0",
    "c_wing_a": "#CCBBBB",    # desaturated rose
    "c_wing_b": "#BBCCD4",    # desaturated blue
}


# =============================================================================
# Layout constants
# =============================================================================

LAYOUT: Dict[str, float] = {
    # X positions (columns) - symmetric around center
    "x_center": 50.0,
    "column_offset": 22.0,        # Distance from center to main columns
    "x_excl_l": 3.0,              # Left exclusion
    "x_excl_r": 97.0,             # Right exclusion

    # Y positions (rows, top to bottom)
    "y_top": 92.0,                # Top box (Presented to ED)
    "y_split": 84.0,              # Split bar
    "y_wing": 73.0,               # Wing A/B boxes
    "y_excl_offset": 0.0,         # Exclusion boxes aligned to wing midline
    "y_analyzed": 52.0,           # Analyzed boxes
    "y_subgroup_bar": 42.0,       # Subgroup split bar
    "y_subgroup": 30.0,           # Subgroup boxes

    # Box dimensions
    "top_w": 24, "top_h": 9,
    "wing_w": 26, "wing_h": 12,
    "analyzed_w": 22, "analyzed_h": 9,
    "subgroup_w": 18, "subgroup_h": 9,
    "excl_w": 12, "excl_h": 7,

    # Subgroup X offsets from x_left
    "subgroup_offset": 11,

    # Zone padding
    "itt_padding": 5.0,
    "pp_pad": 3.0,

    # Margins
    "margin_left": 4.0,
    "margin_right": 3.0,
    "margin_top": 3.0,
    "margin_bottom": 3.0,
}


# =============================================================================
# Geometry helpers
# =============================================================================


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    w: float
    h: float
    face: str
    text: str
    fontsize: int
    fontweight: str = "normal"
    text_line2_bold: bool = False
    sample_size_smaller: bool = False

    def bounds(self) -> Tuple[float, float, float, float]:
        x0 = self.cx - self.w / 2
        y0 = self.cy - self.h / 2
        x1 = self.cx + self.w / 2
        y1 = self.cy + self.h / 2
        return x0, y0, x1, y1

    def anchor(self, side: str) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bounds()
        if side == "top":
            return (self.cx, y1)
        if side == "bottom":
            return (self.cx, y0)
        if side == "left":
            return (x0, self.cy)
        if side == "right":
            return (x1, self.cy)
        raise ValueError(f"Unknown side: {side}")


def draw_box(ax: Axes, box: Box) -> None:
    x0, y0, _, _ = box.bounds()
    rect = FancyBboxPatch(
        (x0, y0),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.01,rounding_size={STYLE['box_rounding']}",
        linewidth=float(STYLE["stroke"]),
        edgecolor=str(STYLE["c_border"]),
        facecolor=box.face,
        zorder=5,
    )
    ax.add_patch(rect)

    if box.text_line2_bold:
        # Draw multi-line text with line 2 in bold.
        lines = box.text.split('\n')
        usable_height = box.h * 0.40
        line_spacing = usable_height / max(len(lines) - 1, 1)
        total_height = (len(lines) - 1) * line_spacing
        start_y = box.cy + total_height / 2

        for i, line in enumerate(lines):
            y_pos = start_y - i * line_spacing
            weight = "bold" if i == 1 else "normal"
            fs = int(STYLE["fs_sample_size"]) if (i == len(lines) - 1) else box.fontsize
            ax.text(
                box.cx,
                y_pos,
                line,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=weight,
                color=str(STYLE["c_text"]),
                zorder=6,
            )
    elif box.sample_size_smaller:
        # Regular text, with a smaller font for the sample-size line.
        lines = box.text.split('\n')
        usable_height = box.h * 0.25
        line_spacing = usable_height / max(len(lines) - 1, 1)
        total_height = (len(lines) - 1) * line_spacing
        start_y = box.cy + total_height / 2

        for i, line in enumerate(lines):
            y_pos = start_y - i * line_spacing
            fs = int(STYLE["fs_sample_size"]) if (line.startswith("(n=") or line.startswith("n=")) else box.fontsize
            ax.text(
                box.cx,
                y_pos,
                line,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=box.fontweight,
                color=str(STYLE["c_text"]),
                zorder=6,
            )
    else:
        ax.text(
            box.cx,
            box.cy,
            box.text,
            ha="center",
            va="center",
            fontsize=box.fontsize,
            fontweight=box.fontweight,
            color=str(STYLE["c_text"]),
            zorder=6,
            linespacing=1.15,
        )


def draw_arrow(
    ax: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    lw: Optional[float] = None,
    major: bool = False,
) -> None:
    """Draw an arrow. If major=True, use a heavier stroke for the main flow."""
    stroke = float(STYLE["stroke_major"]) if major else float(lw if lw is not None else STYLE["stroke"])
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=str(STYLE["arrowstyle"]),
            lw=stroke,
            color=str(STYLE["c_border"]),
            shrinkA=float(STYLE["arrow_shrink"]),
            shrinkB=float(STYLE["arrow_shrink"]),
            mutation_scale=10,
        ),
        zorder=4,
    )


def draw_bar(ax: Axes, x0: float, x1: float, y: float, major: bool = False) -> None:
    """Draw a horizontal bar. If major=True, use a heavier stroke."""
    stroke = float(STYLE["stroke_major"]) if major else float(STYLE["stroke"])
    ax.plot(
        [x0, x1],
        [y, y],
        color=str(STYLE["c_border"]),
        lw=stroke,
        solid_capstyle="butt",
        zorder=3,
    )


def draw_line(ax: Axes, start: Tuple[float, float], end: Tuple[float, float], major: bool = False) -> None:
    """Draw a simple line without an arrowhead. If major=True, use a heavier stroke."""
    stroke = float(STYLE["stroke_major"]) if major else float(STYLE["stroke"])
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=str(STYLE["c_border"]),
        lw=stroke,
        solid_capstyle="butt",
        zorder=3,
    )


def draw_rounded_step_boundary(
    ax: Axes,
    points: list,
    linestyle: str = "--",
    linewidth: float = 1.5,
    edgecolor: str = "#333333",
) -> None:
    """Draw a step-shaped dashed boundary with crisp angular corners."""
    from matplotlib.patches import Polygon

    poly = Polygon(
        points,
        closed=True,
        fill=False,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=edgecolor,
        joinstyle="miter",
        capstyle="butt",
        zorder=1,
    )
    ax.add_patch(poly)


# =============================================================================
# Figure construction
# =============================================================================


def create_figure_1_consort_png(
    *,
    dpi: int = 600,
    out_path: Optional[str] = None,
    preview: bool = False,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ==========================================================================
    # LAYOUT PARAMETERS
    # ==========================================================================

    x_center = LAYOUT["x_center"]
    column_offset = LAYOUT["column_offset"]
    x_left = x_center - column_offset   # Wing A / Analyzed A center
    x_right = x_center + column_offset  # Wing B / Analyzed B center
    x_excl_l = LAYOUT["x_excl_l"]
    x_excl_r = LAYOUT["x_excl_r"]

    y_top = LAYOUT["y_top"]
    y_split = LAYOUT["y_split"]
    y_wing = LAYOUT["y_wing"]
    y_excl = y_wing + LAYOUT["y_excl_offset"]
    y_analyzed = LAYOUT["y_analyzed"]
    y_subgroup_bar = LAYOUT["y_subgroup_bar"]
    y_subgroup = LAYOUT["y_subgroup"]

    top_w, top_h = LAYOUT["top_w"], LAYOUT["top_h"]
    wing_w, wing_h = LAYOUT["wing_w"], LAYOUT["wing_h"]
    analyzed_w, analyzed_h = LAYOUT["analyzed_w"], LAYOUT["analyzed_h"]
    subgroup_w, subgroup_h = LAYOUT["subgroup_w"], LAYOUT["subgroup_h"]
    excl_w, excl_h = LAYOUT["excl_w"], LAYOUT["excl_h"]

    # ==========================================================================
    # CREATE BOXES
    # ==========================================================================

    top_box = Box(
        cx=x_center, cy=y_top, w=top_w, h=top_h,
        face=str(STYLE["c_grey"]),
        text="Presented to the ED\n(n=1,146)",
        fontsize=int(STYLE["fs_title"]),
        sample_size_smaller=True,
    )

    wing_a = Box(
        cx=x_left, cy=y_wing, w=wing_w, h=wing_h,
        face=str(STYLE["c_wing_a"]),
        text="Wing A (Intervention)\nSHAKED Access\n(n=586)",
        fontsize=int(STYLE["fs_title"]),
        text_line2_bold=True,
    )

    wing_b = Box(
        cx=x_right, cy=y_wing, w=wing_w, h=wing_h,
        face=str(STYLE["c_wing_b"]),
        text="Wing B (Control)\nStandard Care\n(n=560)",
        fontsize=int(STYLE["fs_title"]),
        text_line2_bold=True,
    )

    excl_l = Box(
        cx=x_excl_l, cy=y_excl, w=excl_w, h=excl_h,
        face=str(STYLE["c_grey"]),
        text="Excluded\nn=2 (0.3%)",
        fontsize=int(STYLE["fs_box_small"]),
    )

    excl_r = Box(
        cx=x_excl_r, cy=y_excl, w=excl_w, h=excl_h,
        face=str(STYLE["c_grey"]),
        text="Excluded\nn=6 (1.1%)",
        fontsize=int(STYLE["fs_box_small"]),
    )

    ana_a = Box(
        cx=x_left, cy=y_analyzed, w=analyzed_w, h=analyzed_h,
        face=str(STYLE["c_wing_a"]),
        text="Analyzed\n(n=584)",
        fontsize=int(STYLE["fs_title"]),
        sample_size_smaller=True,
    )

    ana_b = Box(
        cx=x_right, cy=y_analyzed, w=analyzed_w, h=analyzed_h,
        face=str(STYLE["c_wing_b"]),
        text="Analyzed\n(n=554)",
        fontsize=int(STYLE["fs_title"]),
        sample_size_smaller=True,
    )

    subgroup_offset = LAYOUT["subgroup_offset"]
    x_nonusers = x_left - subgroup_offset
    x_users = x_left + subgroup_offset

    non_users = Box(
        cx=x_nonusers, cy=y_subgroup, w=subgroup_w, h=subgroup_h,
        face=str(STYLE["c_wing_a"]),
        text="Non-users\nn=325 (55.7%)",
        fontsize=int(STYLE["fs_box"]),
        sample_size_smaller=True,
    )

    users = Box(
        cx=x_users, cy=y_subgroup, w=subgroup_w, h=subgroup_h,
        face=str(STYLE["c_wing_a"]),
        text="SHAKED Users\nn=259 (44.3%)",
        fontsize=int(STYLE["fs_box"]),
        sample_size_smaller=True,
    )

    all_boxes = [top_box, wing_a, wing_b, excl_l, excl_r, ana_a, ana_b, non_users, users]

    # ==========================================================================
    # BACKGROUND ZONES
    # ==========================================================================

    itt_left = 1.0
    itt_right = 99.0
    itt_padding = LAYOUT["itt_padding"]
    itt_bottom = y_subgroup - subgroup_h / 2 - itt_padding
    itt_top = y_wing - wing_h / 2 - 2

    itt_rect = Rectangle(
        (itt_left, itt_bottom),
        itt_right - itt_left,
        itt_top - itt_bottom,
        linewidth=0.5,
        edgecolor="#CCCCCC",
        facecolor=str(STYLE["c_zone"]),
        zorder=0,
    )
    ax.add_patch(itt_rect)

    ax.text(
        itt_left + 1.5,
        itt_top - 1.5,
        "Intention-To-Treat\n(ITT)",
        fontsize=int(STYLE["fs_zone_label"]),
        fontweight=str(STYLE["fw_zone_label"]),
        ha="left",
        va="top",
        color=str(STYLE["c_text"]),
        zorder=1,
        linespacing=1.1,
    )

    pp_pad = LAYOUT["pp_pad"]

    ana_b_x0, ana_b_y0, ana_b_x1, ana_b_y1 = ana_b.bounds()
    users_x0, users_y0, users_x1, users_y1 = users.bounds()
    non_users_x0, non_users_y0, non_users_x1, non_users_y1 = non_users.bounds()

    pp_bottom = users_y0 - pp_pad
    pp_top = ana_b_y1 + pp_pad
    pp_right = ana_b_x1 + pp_pad

    gap_center = (non_users_x1 + users_x0) / 2
    pp_left_users = gap_center

    pp_step_x = users_x1 + pp_pad
    pp_step_y = users_y1 + pp_pad

    pp_points = [
        (pp_left_users, pp_bottom),
        (pp_left_users, pp_step_y),
        (pp_step_x, pp_step_y),
        (pp_step_x, pp_top),
        (pp_right, pp_top),
        (pp_right, pp_bottom),
        (pp_left_users, pp_bottom),
    ]

    draw_rounded_step_boundary(
        ax,
        pp_points,
        linestyle="--",
        linewidth=float(STYLE["stroke_heavy"]),
        edgecolor=str(STYLE["c_border"]),
    )

    ax.text(
        pp_right - 2.0,
        pp_bottom + 1.2,
        "Per-Protocol\n(PP)",
        fontsize=int(STYLE["fs_zone_label"]),
        fontweight=str(STYLE["fw_zone_label"]),
        ha="right",
        va="bottom",
        color=str(STYLE["c_text"]),
        zorder=1,
        linespacing=1.1,
    )

    # ==========================================================================
    # DRAW BOXES
    # ==========================================================================

    for box in all_boxes:
        draw_box(ax, box)

    # ==========================================================================
    # DRAW CONNECTORS
    # ==========================================================================

    draw_line(ax, top_box.anchor("bottom"), (x_center, y_split), major=True)
    draw_bar(ax, x_left, x_right, y_split, major=True)
    draw_arrow(ax, (x_left, y_split), wing_a.anchor("top"), major=True)
    draw_arrow(ax, (x_right, y_split), wing_b.anchor("top"), major=True)

    draw_arrow(ax, wing_a.anchor("left"), excl_l.anchor("right"))
    draw_arrow(ax, wing_b.anchor("right"), excl_r.anchor("left"))

    draw_arrow(ax, wing_a.anchor("bottom"), ana_a.anchor("top"), major=True)
    draw_arrow(ax, wing_b.anchor("bottom"), ana_b.anchor("top"), major=True)

    draw_line(ax, ana_a.anchor("bottom"), (x_left, y_subgroup_bar), major=True)
    draw_bar(ax, x_nonusers, x_users, y_subgroup_bar, major=True)
    draw_arrow(ax, (x_nonusers, y_subgroup_bar), non_users.anchor("top"), major=True)
    draw_arrow(ax, (x_users, y_subgroup_bar), users.anchor("top"), major=True)

    # ==========================================================================
    # PREVIEW MODE: grid overlay for alignment debugging
    # ==========================================================================

    if preview:
        for x in range(0, 101, 10):
            ax.axvline(x=x, color='blue', alpha=0.2, linewidth=0.5, zorder=100)
        for y in range(0, 101, 10):
            ax.axhline(y=y, color='blue', alpha=0.2, linewidth=0.5, zorder=100)
        for x in range(0, 101, 20):
            ax.text(x, 2, str(x), fontsize=6, color='blue', alpha=0.5, ha='center', zorder=100)
        for y in range(0, 101, 20):
            ax.text(2, y, str(y), fontsize=6, color='blue', alpha=0.5, va='center', zorder=100)

    # ==========================================================================
    # AUTO-FIT EXTENTS
    # ==========================================================================

    all_x = []
    all_y = []
    for box in all_boxes:
        x0, y0, x1, y1 = box.bounds()
        all_x.extend([x0, x1])
        all_y.extend([y0, y1])

    all_x.extend([itt_left, itt_right])
    all_y.extend([itt_bottom, itt_top])
    all_x.extend([p[0] for p in pp_points])
    all_y.extend([p[1] for p in pp_points])

    margin_left = LAYOUT["margin_left"]
    margin_right = LAYOUT["margin_right"]
    margin_top = LAYOUT["margin_top"]
    margin_bottom = LAYOUT["margin_bottom"]

    x_min_content = min(all_x) - margin_left
    x_max_content = max(all_x) + margin_right
    y_min_content = min(all_y) - margin_bottom
    y_max_content = max(all_y) + margin_top

    ax.set_xlim(x_min_content, x_max_content)
    ax.set_ylim(y_min_content, y_max_content)

    content_width = x_max_content - x_min_content
    content_height = y_max_content - y_min_content
    aspect = content_width / content_height
    fig_height = 8.0
    fig_width = fig_height * aspect
    fig.set_size_inches(fig_width, fig_height)

    # ==========================================================================
    # EXPORT
    # ==========================================================================

    # Repo root resolved from this file's location (figures/ -> repo root).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)

    if out_path is None:
        base_path = os.path.join(out_dir, "Figure1_CONSORT")
    else:
        if not os.path.isabs(out_path):
            out_path = os.path.join(out_dir, out_path)
        root, ext = os.path.splitext(out_path)
        base_path = root if ext.lower() in {".png", ".pdf"} else out_path

    png_path = f"{base_path}.png"
    pdf_path = f"{base_path}.pdf"

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(png_path, dpi=dpi, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(pdf_path, facecolor="white", edgecolor="none", format="pdf",
                bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)
    return png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 1 (CONSORT flow diagram).")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI (default: 600).")
    parser.add_argument("--out", type=str, default=None, help="Output filename or path.")
    parser.add_argument("--preview", action="store_true",
                        help="Show a grid overlay for alignment debugging.")
    args = parser.parse_args()

    path = create_figure_1_consort_png(
        dpi=args.dpi,
        out_path=args.out,
        preview=args.preview,
    )
    print(f"Saved Figure 1 (CONSORT) to {path}")
