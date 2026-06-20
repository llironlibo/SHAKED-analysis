"""
Figure 2 (Python) -- forest plot of effect estimates and engagement exposure-level.

Panel A: primary estimates (ITT, PP), sensitivity (excluding Study Week 3), and
the instrumental-variable CACE (Wald and Anderson-Rubin).
Panel B: effect by engagement level (active engagement, technical failure,
passive exposure).

Reproduces: Supp Fig 2A (forest / exposure-level). All plotted values are fixed
from the reported estimates and asserted below.

Inputs : none (values are fixed constants, asserted)
Outputs: results/figures/Figure2_v14_*.png and Figure2_v14.pdf
"""

import sys
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.lines import Line2D
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing package: {e}")
    sys.exit(1)

# Output directory -- repo root resolved from this file's location
# (figures/ -> repo root) so the script runs unchanged after a fresh clone.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE, "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STYLE DEFINITIONS
# ============================================================================

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

COLORS = {
    "primary":   "#333333",   # Unified color for all estimates and CIs
    "zero":      "#AAAAAA",   # zero reference line
    "header":    "#000000",   # group header text
    "rule":      "#CCCCCC",   # separator lines
    "rule_dark": "#999999",   # group separator lines
}

FONTS = {
    "panel_title":  10.5,
    "group_header": 9.5,
    "subheader":    7.5,
    "row_label":    8,
    "data":         8,
    "col_header":   8,
    "axis":         9,
}

MARKER_SIZE = 5.0

GROUP_GAP = 0.4
HEADER_OFFSET = 0.70
SUBHEADER_EXTRA_GAP = 0.30
SUBHEADER_OFFSET = 0.55

# ============================================================================
# DATA (fixed effect estimates)
# ============================================================================

GROUPS = {
    0: "Primary Estimates",
    1: "Sensitivity",
    2: "Instrumental variable",
    3: "Exposure-level"
}

SUBHEADERS = {}

# Dagger footnote marker (Unicode U+2020) for the per-protocol rows, written
# as a pure-ASCII escape so the source stays ASCII while the figure renders it.
DAGGER = "\u2020"
ROWS = [
    # Group 0: Primary Estimates (Panel A start)
    {"label": "Intention-to-treat",              "est": -9.4,  "lo": -19.9, "hi":  1.1, "p": 0.077,  "n1": 584, "n2": 554, "group": 0, "shape": "o", "fill": "full"},
    {"label": "Per-protocol " + DAGGER,          "est": -13.3, "lo": -26.2, "hi": -0.4, "p": 0.043,  "n1": 259, "n2": 554, "group": 0, "shape": "o", "fill": "full"},
    # Group 1: Sensitivity
    {"label": "ITT excl. Week 3",                "est": -20.1, "lo": -38.1, "hi": -2.0, "p": 0.001,  "n1": 444, "n2": 423, "group": 1, "shape": "o", "fill": "full"},
    {"label": "PP excl. Week 3 " + DAGGER,       "est": -24.2, "lo": -48.0, "hi": -0.5, "p": 0.001,  "n1": 206, "n2": 423, "group": 1, "shape": "o", "fill": "full"},
    # Group 2: Instrumental variable
    {"label": "CACE (Wald LATE)",                "est": -32.2, "lo": -65.2, "hi":  0.9, "p": 0.056,  "n1": 259, "n2": 0,   "group": 2, "shape": "D", "fill": "full"},
    {"label": "CACE (Anderson-Rubin)",           "est": -32.2, "lo": -65.6, "hi":  0.9, "p": 0.057,  "n1": 259, "n2": 0,   "group": 2, "shape": "D", "fill": "full"},
    # Group 3: Exposure-level (Panel B)
    {"label": "Active engagement",               "est": -13.6, "lo": -26.4, "hi": -0.4, "p": 0.048,  "n1": 215, "n2": 554, "group": 3, "shape": "o", "fill": "full"},
    {"label": "Technical failure",               "est": -11.5, "lo": -39.1, "hi": 20.9, "p": 0.45,   "n1":  44, "n2": 554, "group": 3, "shape": "o", "fill": "full"},
    {"label": "Passive exposure",                "est": -1.3,  "lo": -14.2, "hi": 12.3, "p": 0.84,   "n1": 273, "n2": 554, "group": 3, "shape": "o", "fill": "full"},
]

# ============================================================================
# ASSERTIONS (verify fixed values)
# ============================================================================

assert ROWS[0]["est"] == -9.4,   "ITT estimate"
assert ROWS[1]["est"] == -13.3,  "PP estimate"
assert ROWS[2]["est"] == -20.1,  "ITT excl Week 3 estimate"
assert ROWS[3]["est"] == -24.2,  "PP excl Week 3 estimate"
assert ROWS[4]["est"] == -32.2,  "CACE Wald estimate"
assert ROWS[5]["est"] == -32.2,  "CACE AR estimate"
assert ROWS[6]["est"] == -13.6,  "Active estimate"
assert ROWS[6]["p"]   == 0.048,  "Active p-value"
assert ROWS[7]["est"] == -11.5,  "TechFailure estimate"
assert ROWS[8]["est"] == -1.3,   "Passive estimate"
assert ROWS[8]["p"]   == 0.84,   "Passive p-value"

print("[OK] Data assertions passed (9 rows, 4 groups)")

# ============================================================================
# HELPERS
# ============================================================================

def format_p(p):
    """Format a p-value with a leading zero for decimal alignment."""
    if p is None:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def compute_layout():
    """Compute exact y positions for all rows, panel elements, and headers."""
    layout = {}
    row_ys = []
    group_header_ys = {}

    y = 0.0
    layout['global_rule_top'] = y
    y += 0.5
    layout['global_headers'] = y
    y += 0.5
    layout['global_rule_bottom'] = y
    y += 0.6

    layout['panel_a_title'] = y
    y += 0.8

    current_group = -1
    for i, row in enumerate(ROWS):
        g = row["group"]

        if g == 3 and current_group != 3:
            # Mirror the Panel A title spacing for the Panel B title.
            y += 0.3
            layout['panel_b_title'] = y
            y += 0.8

        if g != current_group:
            if current_group >= 0 and g != 3:  # Intra-panel gaps
                y += GROUP_GAP
            group_header_ys[g] = y
            y += HEADER_OFFSET
            current_group = g

        row_ys.append(y)
        y += 1.0

    y -= 0.6  # Adjust bottom padding
    layout['panel_b_rule_end'] = y

    return layout, row_ys, group_header_ys


# ============================================================================
# MAIN FIGURE
# ============================================================================

COL_N = 0.12        # N: center-aligned
COL_EST = 0.37      # Effect: right-aligned
COL_CI = 0.62       # 95% CI: center-aligned
COL_P = 0.92        # P Value: right-aligned

def draw_column_headers(ax_label, ax_data, y_pos):
    """Draw column headers at the given Y position for the Label and Data columns."""
    ax_label.text(0.02, y_pos, "Analysis",
                  fontsize=FONTS["col_header"], fontweight="bold",
                  va="center", ha="left",
                  transform=ax_label.get_yaxis_transform())

    ax_data.text(COL_N, y_pos, "N\n(Interv./Control)",
                 fontsize=FONTS["col_header"], fontweight="bold",
                 va="center", ha="center", linespacing=0.95,
                 transform=ax_data.get_yaxis_transform())
    ax_data.text(COL_EST, y_pos, "Effect\n(min)",
                 fontsize=FONTS["col_header"], fontweight="bold",
                 va="center", ha="center", linespacing=0.95,
                 transform=ax_data.get_yaxis_transform())
    ax_data.text(COL_CI, y_pos, "95% CI",
                 fontsize=FONTS["col_header"], fontweight="bold",
                 va="center", ha="center", linespacing=0.95,
                 transform=ax_data.get_yaxis_transform())
    ax_data.text(COL_P, y_pos, "P Value",
                 fontsize=FONTS["col_header"], fontweight="bold",
                 va="center", ha="right", linespacing=0.95,
                 transform=ax_data.get_yaxis_transform())

def draw_header_rules(axes, y_top, y_bottom):
    """Draw sandwich header rules across all axes (thick top, thin bottom)."""
    for ax in axes:
        ax.axhline(y=y_top, color="black", linewidth=1.5,
                   clip_on=False, solid_capstyle="butt")
        ax.axhline(y=y_bottom, color="black", linewidth=0.8,
                   clip_on=False, solid_capstyle="butt")

def create_figure():
    """Create the grouped forest plot."""
    layout, row_ys, group_header_ys = compute_layout()

    fig = plt.figure(figsize=(7.1, 7.8))

    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.33, 0.27, 0.40],
                  wspace=0.0)
    ax_label = fig.add_subplot(gs[0, 0])
    ax_forest = fig.add_subplot(gs[0, 1], sharey=ax_label)
    ax_data = fig.add_subplot(gs[0, 2], sharey=ax_label)

    y_min = layout['global_rule_top'] - 0.5
    y_max = layout['panel_b_rule_end'] + 0.6

    for ax in (ax_label, ax_forest, ax_data):
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()

    all_axes = [ax_label, ax_forest, ax_data]

    # ------------------------------------------------------------------
    # RULES & SEPARATORS
    # ------------------------------------------------------------------
    draw_header_rules(all_axes, layout['global_rule_top'], layout['global_rule_bottom'])

    # ------------------------------------------------------------------
    # COLUMN 1: Labels & Titles
    # ------------------------------------------------------------------
    ax_label.set_xlim(0, 1)
    ax_label.axis("off")

    ax_label.text(0.00, layout['panel_a_title'], "Panel A: Effect Estimates",
                  fontsize=FONTS["panel_title"], fontweight="bold",
                  va="center", ha="left", transform=ax_label.get_yaxis_transform())

    ax_label.text(0.00, layout['panel_b_title'], "Panel B: Effect by Engagement",
                  fontsize=FONTS["panel_title"], fontweight="bold",
                  va="center", ha="left", transform=ax_label.get_yaxis_transform())

    for g, hy in group_header_ys.items():
        ax_label.text(0.03, hy, GROUPS[g],
                      fontsize=FONTS["group_header"], fontweight="bold",
                      va="center", ha="left", transform=ax_label.get_yaxis_transform())

    for i, ry in enumerate(row_ys):
        indent = 0.06
        ax_label.text(indent, ry, ROWS[i]["label"],
                      fontsize=FONTS["row_label"], va="center", ha="left",
                      transform=ax_label.get_yaxis_transform())

    # ------------------------------------------------------------------
    # COLUMN 2: Forest plot
    # ------------------------------------------------------------------
    x_lo, x_hi = -70, 25
    ax_forest.set_xlim(x_lo, x_hi)
    ax_forest.set_xticks([-60, -40, -20, 0, 20])
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["left"].set_visible(False)
    ax_forest.set_yticks([])
    ax_forest.spines["bottom"].set_color("black")
    ax_forest.spines["bottom"].set_linewidth(1.0)
    ax_forest.tick_params(axis="x", labelsize=FONTS["data"], width=1.0,
                          length=4, pad=3)

    ax_forest.plot([0, 0], [layout['global_rule_bottom'], layout['panel_b_rule_end']],
                   color=COLORS["zero"], linewidth=0.8, zorder=1)

    ax_forest.set_xlabel("Consultation time difference (minutes)",
                         fontsize=FONTS["axis"], fontweight="bold",
                         labelpad=6)

    for i, ry in enumerate(row_ys):
        row = ROWS[i]
        est, lo, hi = row["est"], row["lo"], row["hi"]
        color = COLORS["primary"]

        lo_draw = max(lo, x_lo)
        hi_draw = min(hi, x_hi)

        ax_forest.plot([lo_draw, hi_draw], [ry, ry],
                       color=color, linewidth=1.8,
                       linestyle="-", zorder=5,
                       solid_capstyle="round")

        cap_half = 0.15
        if lo >= x_lo:
            ax_forest.plot([lo, lo], [ry - cap_half, ry + cap_half],
                           color=color, linewidth=1.5, zorder=5,
                           solid_capstyle="round")
        if hi <= x_hi:
            ax_forest.plot([hi, hi], [ry - cap_half, ry + cap_half],
                           color=color, linewidth=1.5, zorder=5,
                           solid_capstyle="round")

        marker_edge = color
        marker_face = color if row["fill"] == "full" else "white"

        ax_forest.plot(est, ry, row["shape"],
                       color=color, markerfacecolor=marker_face,
                       markeredgecolor=marker_edge, markeredgewidth=1.2,
                       markersize=MARKER_SIZE, zorder=6)

    # ------------------------------------------------------------------
    # COLUMN 3: Data columns / Headers
    # ------------------------------------------------------------------
    ax_data.set_xlim(0, 1)
    ax_data.axis("off")

    draw_column_headers(ax_label, ax_data, layout['global_headers'])

    for i, ry in enumerate(row_ys):
        row = ROWS[i]

        n1, n2 = row['n1'], row['n2']
        n_str = f"{n1}/{n2}" if n2 > 0 else f"{n1}"
        ax_data.text(COL_N, ry, n_str,
                     fontsize=FONTS["data"], va="center", ha="center", transform=ax_data.get_yaxis_transform())

        est_str = f"{row['est']:.1f}"
        ax_data.text(COL_EST + 0.03, ry, est_str,
                     fontsize=FONTS["data"], va="center", ha="right", transform=ax_data.get_yaxis_transform())

        ci_str = f"{row['lo']:.1f} to {row['hi']:.1f}"
        ax_data.text(COL_CI, ry, ci_str,
                     fontsize=FONTS["data"], va="center", ha="center", transform=ax_data.get_yaxis_transform())

        p_str = format_p(row["p"])
        ax_data.text(COL_P, ry, p_str,
                     fontsize=FONTS["data"], va="center", ha="right",
                     color="#000000", fontweight="normal", transform=ax_data.get_yaxis_transform())

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["primary"],
               markersize=MARKER_SIZE, label='Median based estimate (HL)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS["primary"],
               markersize=MARKER_SIZE, label='Mean based estimate')
    ]
    ax_data.legend(handles=legend_elements, loc='upper right', ncol=1,
                   frameon=False, edgecolor='none', framealpha=1,
                   fontsize=FONTS["row_label"], borderaxespad=0.0,
                   bbox_to_anchor=(COL_P, -0.05))

    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.18)

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    print(f"[INFO] Figure 2 run mode: {mode}")

    fig = create_figure()

    base = os.path.join(OUTPUT_DIR, "Figure2_v14")

    if mode == "preview":
        path = f"{base}_preview.png"
        fig.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {path}")
    elif mode == "final":
        for dpi in [300, 600]:
            path = f"{base}_{dpi}dpi.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"Saved: {path}")
        pdf_path = f"{base}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {pdf_path}")
    else:
        print(f"Unknown mode: {mode}. Use 'preview' or 'final'.")
        sys.exit(1)

    plt.close(fig)
    print(f"[DONE] Figure 2 complete (mode: {mode})")


if __name__ == "__main__":
    main()
