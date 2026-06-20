# =============================================================================
# _common.R -- Shared infrastructure for the SHAKED manuscript figures.
# Source this at the top of every figure script:
#   source(file.path(dirname(sys.frame(1)$ofile), "_common.R"))   # when sourced
# In practice the figure scripts resolve this file next to themselves.
#
# Requires: R 4.5+, ggplot2 4.0+
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(ggtext)  # element_markdown() in labels
  library(ragg)    # system fonts (Arial) natively -- no showtext needed
  library(dplyr)
  library(readr)
})

# Note: showtext is intentionally not used -- it conflicts with ragg::agg_png
# and can produce invisible text in PNG output. ragg resolves Arial via the
# system fonts automatically.

# -- Color palette -----------------------------------------------------------
pal <- list(
  active   = "#4472C4", # Steel blue -- SHAKED / significant / active engagement
  control  = "#A5A5A5", # Medium gray -- control / reference
  null     = "#D9D9D9", # Light gray -- null findings / passive / tech failure
  anomaly  = "#E07B54", # Muted coral -- Study Week 3 (anomalous period)
  emphasis = "#2F5496", # Dark navy -- causal emphasis / fitted curves
  bg_coral = "#FDE8E0", # Very light coral -- Study Week 3 zone background
  ci_band  = "#D6E4F0", # Light blue -- confidence band fill
  text     = "#333333"  # Near-black -- primary text
)

# -- Custom theme (ggplot2 4.0: ink/paper/accent) ----------------------------
theme_shaked <- function(base_size = 10) {
  theme_minimal(
    base_family = "Arial",
    base_size = base_size,
    ink = pal$text,
    paper = "white"
  ) +
    theme(
      geom = element_geom(
        ink = pal$text,
        accent = pal$active,
        linewidth = 0.5,
        borderwidth = 0.3
      )
    ) +
    theme_sub_panel(
      grid.minor = element_blank(),
      grid.major.y = element_blank()
    ) +
    theme_sub_axis(
      line = element_line(linewidth = 0.3),
      ticks = element_line(linewidth = 0.3)
    ) +
    theme_sub_plot(
      title = element_text(face = "bold", size = base_size + 2),
      subtitle = element_text(color = "gray40"),
      tag = element_text(face = "bold", size = 14),
      margin = margin_auto(5, 5)
    ) +
    theme(
      legend.position = "none",  # figures use direct annotation, not legends
      palette.shape.discrete = c(
        "circle", "diamond", "diamond open", "triangle"
      )
    )
}

# -- P-value formatter -------------------------------------------------------
format_p <- function(p) {
  ifelse(p < 0.001, "<.001",
    ifelse(p >= 0.995, ">.99",
      sprintf(".%03d", round(p * 1000))
    )
  )
}
# Edge case: p = 1.0 -> ">.99" (can occur with Holm-adjusted p-values)

# -- CI label formatter ------------------------------------------------------
format_ci <- function(est, lo, hi) {
  sprintf("%.1f [%.1f, %.1f]", est, lo, hi)
}

# -- Repo-root resolution + save helper --------------------------------------
# The figure scripts live in figures/, one level below the repo root. Recover
# this file's own path (whether sourced or run via Rscript) and walk up one
# level to the repo root, so figures build after a fresh clone.
.shaked_repo_root <- function() {
  this_file <- NULL
  # When sourced, sys.frames() / ofile holds the path to _common.R.
  for (i in seq_len(sys.nframe())) {
    of <- get0("ofile", envir = sys.frame(i), inherits = FALSE)
    if (!is.null(of) && nzchar(of)) { this_file <- of; break }
  }
  # When run via Rscript directly, recover from the --file= command argument.
  if (is.null(this_file)) {
    fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(fa) > 0) this_file <- sub("^--file=", "", fa[1])
  }
  if (!is.null(this_file) && nzchar(this_file)) {
    return(normalizePath(file.path(dirname(this_file), ".."), mustWork = FALSE))
  }
  normalizePath(getwd(), mustWork = FALSE)
}
BASE_DIR <- .shaked_repo_root()
FIG_DIR <- file.path(BASE_DIR, "results", "figures")
TEMP_DIR <- file.path(FIG_DIR, "temp")
DATA_DIR <- file.path(BASE_DIR, "data", "cleaned")
ADOPTION_CSV <- file.path(BASE_DIR, "results", "eFigure3_Adoption_TimeSeries_data.csv")
IPTW_CSV <- file.path(BASE_DIR, "results", "eFigure4_IPTW_Balance_data.csv")

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TEMP_DIR, showWarnings = FALSE, recursive = TRUE)

save_figure <- function(plot, name, width, height, dpi = 600) {
  png_path <- file.path(FIG_DIR, paste0(name, "_600dpi.png"))
  pdf_path <- file.path(FIG_DIR, paste0(name, ".pdf"))
  ggsave(png_path, plot,
    width = width, height = height,
    dpi = dpi, device = ragg::agg_png
  )
  ggsave(pdf_path, plot,
    width = width, height = height,
    device = cairo_pdf
  )
  cat("Saved:", png_path, "\n")
  cat("Saved:", pdf_path, "\n")
}

# -- Typography constants (Nature-family: 7pt data, 8pt headers) -------------
# ggplot2 size in mm; 1pt = 0.353mm
FONT_SIZES <- list(
  data       = 2.5,  # 7pt -- data labels, row text
  header     = 2.8,  # 8pt -- section headers, column headers
  stats      = 2.3,  # 6.5pt -- CI text, p-values
  annotation = 2.1,  # 6pt -- footnotes, direction labels
  axis       = 2.5,  # 7pt -- axis labels
  axis_tick  = 2.1,  # 6pt -- axis tick labels
  tag        = 4.2   # 12pt -- panel tags (A, B)
)

# -- Sample-size-proportional point scaling ----------------------------------
scale_pt_size <- function(n_vec, base = 2.5, range = 2.5) {
  base + range * sqrt(n_vec / max(n_vec))
}

# -- Section band layers -----------------------------------------------------
make_section_bands <- function(band_ranges) {
  layers <- list()
  for (b in band_ranges) {
    layers <- c(layers, list(
      annotate("rect",
        xmin = -Inf, xmax = Inf,
        ymin = b$ymin, ymax = b$ymax,
        fill = b$fill, alpha = 1
      )
    ))
  }
  layers
}

cat("[_common.R] SHAKED figure infrastructure loaded.\n")
cat("[_common.R] ggplot2", as.character(packageVersion("ggplot2")), "\n")
