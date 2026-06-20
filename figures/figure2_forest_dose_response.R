# =============================================================================
# figure2_forest_dose_response.R -- SHAKED manuscript Figure 2 (R version)
#
# Panel A: Causal estimation hierarchy (ITT / PP / CACE)
#   - Uniform neutral styling; no significance encoding
#   - 3 rows, all in minutes (HL / Wald estimates)
#
# Panel B: Engagement-level mechanism (exposure-level)
#   - Significance-encoded: null -> null -> significant staircase
#   - 4 rows (Wing B reference + 3 engagement levels), HL shifts vs Wing B
#
# Reproduces: Supp Fig 2A (forest / exposure-level). Values are fixed and asserted.
# Output: results/figures/Figure2_Forest_DoseResponse_v3_600dpi.png + .pdf
# Size: 180mm x ~84mm
# =============================================================================

# Resolve _common.R next to this script (portable across CWDs).
.this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
source(file.path(dirname(.this_file), "_common.R"))

# Run modes: preview | final | both (default)
args <- commandArgs(trailingOnly = TRUE)
run_mode <- if (length(args) >= 1) tolower(args[1]) else "both"
if (!run_mode %in% c("preview", "final", "both")) {
    stop("Invalid mode. Use: preview, final, or both")
}
message(sprintf("[INFO] Figure 2 run mode: %s", run_mode))

# =============================================================================
# 1. DATA -- fixed effect estimates
# =============================================================================

# Panel A: 3 causal estimation approaches (all minutes)
panel_a_data <- data.frame(
    label    = c("Intention-to-treat", "Per-protocol", "CACE (IV)"),
    estimate = c(-9.4, -13.3, -32.2),
    ci_low   = c(-19.9, -26.2, -65.2),
    ci_high  = c(1.1, -0.4, 0.9),
    p_value  = c(0.077, 0.043, 0.056),
    n_tx     = c(584, 259, 584),
    n_ctrl   = c(554, 554, 554),
    y_pos    = c(2.4, 1.7, 1),
    is_ref   = c(FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
)

# Panel B: 4 engagement levels (HL shifts vs Wing B reference)
panel_b_data <- data.frame(
    label    = c("Wing B controls (reference)", "Passive exposure",
                 "Technical failure", "Active engagement"),
    estimate = c(0, -1.3, -11.5, -13.6),
    ci_low   = c(NA, -14.2, -39.1, -26.4),
    ci_high  = c(NA, 12.3, 20.9, -0.4),
    p_value  = c(NA, 0.844, 0.449, 0.048),
    n_tx     = c(NA, 273, 44, 215),
    n_ctrl   = c(554, 554, 554, 554),
    y_pos    = c(3.1, 2.4, 1.7, 1),
    is_ref   = c(TRUE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
)

# =============================================================================
# 2. ASSERTIONS
# =============================================================================
stopifnot(nrow(panel_a_data) == 3)
stopifnot(nrow(panel_b_data) == 4)
stopifnot(panel_a_data$estimate[1] == -9.4)    # ITT
stopifnot(panel_a_data$estimate[2] == -13.3)   # PP
stopifnot(panel_a_data$estimate[3] == -32.2)   # CACE
stopifnot(panel_a_data$p_value[2]  == 0.043)   # PP
stopifnot(panel_a_data$p_value[3]  == 0.056)   # CACE
stopifnot(panel_b_data$estimate[4] == -13.6)   # Active
stopifnot(panel_b_data$p_value[4]  == 0.048)   # Active significant
stopifnot(panel_b_data$is_ref[1]   == TRUE)    # Wing B is reference
message("[OK] Figure 2 data assertions passed")

# =============================================================================
# 3. DERIVED COLUMNS
# =============================================================================

FS <- FONT_SIZES
sig_color     <- pal$active   # "#4472C4"
nsig_color    <- "#555555"    # dark gray markers for non-significant
nsig_ci_color <- "#AAAAAA"   # lighter gray CI lines for non-significant
ref_color     <- pal$control  # "#A5A5A5"
neutral_color <- "#333333"    # Panel A uniform color
header_rule_y <- 3.55         # y-position for header rule in Panel A
row_sp        <- 1            # row spacing unit

# --- Panel A derived ---
panel_a_data$pt_color  <- neutral_color
panel_a_data$n_display <- paste0(panel_a_data$n_tx, "/", panel_a_data$n_ctrl)
panel_a_data$est_text  <- sprintf("%.1f", panel_a_data$estimate)
panel_a_data$ci_text   <- sprintf("%.1f to %.1f",
    panel_a_data$ci_low, panel_a_data$ci_high)
panel_a_data$p_text    <- format_p(panel_a_data$p_value)
panel_a_data$p_color   <- neutral_color  # uniform, no significance encoding

# --- Panel B derived ---
panel_b_data$significant <- !panel_b_data$is_ref &
    !is.na(panel_b_data$p_value) & panel_b_data$p_value < 0.05

panel_b_data$pt_color <- ifelse(panel_b_data$is_ref, ref_color,
    ifelse(panel_b_data$significant, sig_color, nsig_color))

panel_b_data$n_display <- ifelse(panel_b_data$is_ref,
    as.character(panel_b_data$n_ctrl),
    paste0(panel_b_data$n_tx, "/", panel_b_data$n_ctrl))

panel_b_data$est_text <- ifelse(panel_b_data$is_ref, "",
    sprintf("%.1f", panel_b_data$estimate))
panel_b_data$ci_text  <- ifelse(panel_b_data$is_ref, "",
    sprintf("%.1f to %.1f", panel_b_data$ci_low, panel_b_data$ci_high))
panel_b_data$p_text   <- ifelse(panel_b_data$is_ref, "",
    format_p(panel_b_data$p_value))
panel_b_data$p_color  <- ifelse(panel_b_data$significant, sig_color, nsig_color)
panel_b_data$p_color[panel_b_data$is_ref] <- nsig_color

panel_b_ci <- panel_b_data[!panel_b_data$is_ref, ]

# Shared x-axis range
x_limits <- c(-75, 25)
x_breaks <- seq(-60, 20, 20)

# =============================================================================
# Helper: build one panel (6-column table-style forest plot)
# =============================================================================
build_panel <- function(df, df_ci, panel_tag, y_range, header_y, rule_y,
                        show_x_axis = FALSE, uniform_style = FALSE) {

    # --- Column 1: Labels ---
    col_labels <- ggplot(df, aes(x = 0, y = y_pos)) +
        geom_text(aes(label = label), hjust = 0, size = FS$data,
            family = "Arial", color = pal$text) +
        annotate("text", x = -0.02, y = header_y + 0.15, label = panel_tag,
            hjust = 0, fontface = "bold", size = FS$tag, family = "Arial",
            color = pal$text) +
        annotate("segment", x = -0.02, xend = 1.1, y = rule_y, yend = rule_y,
            linewidth = 0.3, color = "gray70") +
        scale_y_continuous(limits = y_range, expand = c(0, 0)) +
        scale_x_continuous(limits = c(-0.02, 1)) +
        theme_void() +
        theme(plot.margin = margin(2, 0, 2, 4))

    # --- Column 2: N (Interv./Control) ---
    col_n <- ggplot(df, aes(x = 0.5, y = y_pos)) +
        geom_text(aes(label = n_display), hjust = 0.5, size = FS$stats,
            family = "Arial", color = pal$text) +
        annotate("text", x = 0.5, y = header_y,
            label = "N\n(Tx/Ctrl)",
            hjust = 0.5, fontface = "bold", size = FS$annotation,
            family = "Arial", color = pal$text, lineheight = 0.9) +
        annotate("segment", x = -0.3, xend = 1.3, y = rule_y, yend = rule_y,
            linewidth = 0.3, color = "gray70") +
        scale_y_continuous(limits = y_range, expand = c(0, 0)) +
        scale_x_continuous(limits = c(-0.3, 1.3)) +
        theme_void() +
        theme(plot.margin = margin(2, 0, 2, 0))

    # --- Column 3: CI forest plot ---
    p_ci <- ggplot() +
        geom_vline(xintercept = 0, color = "gray70", linewidth = 0.5)

    if (any(df$is_ref %in% TRUE)) {
        p_ci <- p_ci +
            geom_point(data = df[df$is_ref, ],
                aes(x = estimate, y = y_pos),
                shape = 18, size = 3.5, color = ref_color)
    }

    if (!is.null(df_ci) && nrow(df_ci) > 0) {
        if (uniform_style) {
            # Panel A: all same color, solid lines, filled squares
            p_ci <- p_ci +
                geom_segment(data = df_ci,
                    aes(x = ci_low, xend = ci_high, y = y_pos, yend = y_pos),
                    linewidth = 1.1, color = neutral_color) +
                geom_point(data = df_ci,
                    aes(x = estimate, y = y_pos),
                    shape = 15, size = 3.2, color = neutral_color)
        } else {
            # Panel B: significance-encoded
            nsig_rows <- df_ci[!df_ci$significant, ]
            sig_rows  <- df_ci[df_ci$significant, ]

            if (nrow(nsig_rows) > 0) {
                p_ci <- p_ci +
                    geom_segment(data = nsig_rows,
                        aes(x = ci_low, xend = ci_high,
                            y = y_pos, yend = y_pos),
                        linewidth = 1.0, color = nsig_ci_color) +
                    geom_point(data = nsig_rows,
                        aes(x = estimate, y = y_pos),
                        shape = 15, size = 3.2, color = nsig_color)
            }
            if (nrow(sig_rows) > 0) {
                p_ci <- p_ci +
                    geom_segment(data = sig_rows,
                        aes(x = ci_low, xend = ci_high,
                            y = y_pos, yend = y_pos),
                        linewidth = 1.2, color = sig_color) +
                    geom_point(data = sig_rows,
                        aes(x = estimate, y = y_pos),
                        shape = 15, size = 3.5, color = sig_color)
            }
        }
    }

    p_ci <- p_ci +
        annotate("segment",
            x = x_limits[1], xend = x_limits[2],
            y = rule_y, yend = rule_y,
            linewidth = 0.3, color = "gray70")

    # Direction arrows (only on the bottom panel)
    if (show_x_axis) {
        p_ci <- p_ci +
            annotate("text", x = -38, y = y_range[1] + 0.35,
                label = "\u2190 Favors SHAKED",
                hjust = 0.5, size = FS$annotation, color = "gray50",
                fontface = "italic", family = "Arial") +
            annotate("text", x = 12, y = y_range[1] + 0.35,
                label = "Favors Control \u2192",
                hjust = 0.5, size = FS$annotation, color = "gray50",
                fontface = "italic", family = "Arial")
    }

    p_ci <- p_ci +
        scale_x_continuous(limits = x_limits, breaks = x_breaks) +
        scale_y_continuous(limits = y_range, expand = c(0, 0)) +
        labs(x = if (show_x_axis) "Consultation Time Difference (minutes)"
             else NULL, y = NULL) +
        theme_shaked(base_size = 8) +
        theme(
            axis.text.y  = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x  = if (show_x_axis)
                element_text(size = FS$axis_tick * 2.83)
                else element_blank(),
            axis.ticks.x = if (show_x_axis)
                element_line(linewidth = 0.3)
                else element_blank(),
            axis.title.x = if (show_x_axis)
                element_text(size = FS$axis * 2.83)
                else element_blank(),
            panel.grid.major.x = element_line(color = "gray94", linewidth = 0.2),
            panel.grid.major.y = element_blank(),
            plot.margin = margin(2, 2, if (show_x_axis) 5 else 2, 2)
        )

    # --- Column 4: Median diff. ---
    col_est <- ggplot(df, aes(x = 0.5, y = y_pos)) +
        geom_text(aes(label = est_text), hjust = 0.5, size = FS$stats,
            family = "Arial", color = pal$text) +
        annotate("text", x = 0.5, y = header_y,
            label = "Median\ndiff.",
            hjust = 0.5, fontface = "bold", size = FS$annotation,
            family = "Arial", color = pal$text, lineheight = 0.9) +
        annotate("segment", x = -0.3, xend = 1.3, y = rule_y, yend = rule_y,
            linewidth = 0.3, color = "gray70") +
        scale_y_continuous(limits = y_range, expand = c(0, 0)) +
        scale_x_continuous(limits = c(-0.3, 1.3)) +
        theme_void() +
        theme(plot.margin = margin(2, 0, 2, 0))

    # --- Column 5: 95% CI ---
    col_ci_text <- ggplot(df, aes(x = 0.5, y = y_pos)) +
        geom_text(aes(label = ci_text), hjust = 0.5, size = FS$stats,
            family = "Arial", color = pal$text) +
        annotate("text", x = 0.5, y = header_y,
            label = "95% CI",
            hjust = 0.5, fontface = "bold", size = FS$annotation,
            family = "Arial", color = pal$text) +
        annotate("segment", x = -0.3, xend = 1.3, y = rule_y, yend = rule_y,
            linewidth = 0.3, color = "gray70") +
        scale_y_continuous(limits = y_range, expand = c(0, 0)) +
        scale_x_continuous(limits = c(-0.3, 1.3)) +
        theme_void() +
        theme(plot.margin = margin(2, 0, 2, 0))

    # --- Column 6: P Value ---
    col_p <- ggplot(df, aes(x = 0.5, y = y_pos)) +
        geom_text(aes(label = p_text), hjust = 1, size = FS$stats,
            family = "Arial", color = df$p_color) +
        annotate("text", x = 0.5, y = header_y,
            label = "P Value",
            hjust = 1, fontface = "bold", size = FS$annotation,
            family = "Arial", color = pal$text) +
        annotate("segment", x = -0.5, xend = 0.6, y = rule_y, yend = rule_y,
            linewidth = 0.3, color = "gray70") +
        scale_y_continuous(limits = y_range, expand = c(0, 0)) +
        scale_x_continuous(limits = c(-0.5, 0.6)) +
        theme_void() +
        theme(plot.margin = margin(2, 6, 2, 0))

    # Assemble: 6 columns
    (col_labels | col_n | p_ci | col_est | col_ci_text | col_p) +
        plot_layout(widths = c(3.2, 1.2, 2.8, 0.8, 1.6, 0.8))
}

# =============================================================================
# 4. PANEL A -- Uniform styling (no significance encoding)
# =============================================================================

y_range_a  <- c(0.6, 3.15)
header_y_a <- 3.0
rule_y_a   <- 2.88

panel_a <- build_panel(
    df            = panel_a_data,
    df_ci         = panel_a_data,
    panel_tag     = "A",
    y_range       = y_range_a,
    header_y      = header_y_a,
    rule_y        = rule_y_a,
    show_x_axis   = FALSE,
    uniform_style = TRUE
)

# =============================================================================
# 5. PANEL B -- Significance-encoded engagement staircase
# =============================================================================

y_range_b  <- c(0.4, 3.85)
header_y_b <- 3.7
rule_y_b   <- 3.58

panel_b <- build_panel(
    df            = panel_b_data,
    df_ci         = panel_b_ci,
    panel_tag     = "B",
    y_range       = y_range_b,
    header_y      = header_y_b,
    rule_y        = rule_y_b,
    show_x_axis   = TRUE,
    uniform_style = FALSE
)

# =============================================================================
# 6. ASSEMBLY -- stacked vertically, ~40/60 height split
# =============================================================================

fig2 <- (panel_a / panel_b) +
    plot_layout(heights = c(0.40, 0.60))

# =============================================================================
# 7. SAVE
# =============================================================================
FIG_W <- 7.1   # 180mm
FIG_H <- 3.3   # ~84mm

save_preview <- function(plot, name, width, height, dpi = 180) {
    png_path <- file.path(FIG_DIR, paste0(name, "_preview.png"))
    ggsave(
        png_path, plot,
        width = width, height = height, dpi = dpi, device = ragg::agg_png
    )
    cat("Saved:", png_path, "\n")
}

if (run_mode %in% c("preview", "both")) {
    save_preview(fig2, "Figure2_Forest_DoseResponse_v3", width = FIG_W, height = FIG_H)
}
if (run_mode %in% c("final", "both")) {
    save_figure(fig2, "Figure2_Forest_DoseResponse_v3", width = FIG_W, height = FIG_H)
}
cat(sprintf("[DONE] Figure 2 complete (mode: %s).\n", run_mode))
