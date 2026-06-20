# =============================================================================
# figure3_adoption_dynamics.R
# SHAKED manuscript Figure 3: Adoption dynamics + weekly ITT effects.
#
# Panel A: Daily adoption with logistic decay (date x-axis)
# Panel B: Weekly ITT bars (categorical x-axis)
#
# Reproduces: Figure 3 (adoption dynamics).
# Input : results/eFigure3_Adoption_TimeSeries_data.csv  (written by stage 12)
# Output: results/figures/Figure3_Adoption_Dynamics_600dpi.png + .pdf
# Size: 7 x 6 inches
# =============================================================================

# Resolve _common.R next to this script (portable across CWDs).
.this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
source(file.path(dirname(.this_file), "_common.R"))

# -- 0. Force English locale for date labels ---------------------------------
old_locale <- Sys.getlocale("LC_TIME")
try(Sys.setlocale("LC_TIME", "English_United States.1252"), silent = TRUE)
on.exit(try(Sys.setlocale("LC_TIME", old_locale), silent = TRUE), add = TRUE)

# -- 1. Panel A data -- daily adoption from CSV ------------------------------

df <- read_csv(ADOPTION_CSV, show_col_types = FALSE)
df$date <- as.Date(df$arrival_date)

# -- 2. Panel B data -- weekly ITT effects (fixed from reported values) ------

weekly_effects <- data.frame(
  week_label = factor(
    c("Week 1\n(Nov 9-16)", "Week 2\n(Nov 17-23)",
      "Week 3\n(Nov 24-30)", "Week 4\n(Dec 1-7)"),
    levels = c("Week 1\n(Nov 9-16)", "Week 2\n(Nov 17-23)",
               "Week 3\n(Nov 24-30)", "Week 4\n(Dec 1-7)")
  ),
  n_a = c(159, 137, 140, 148),
  n_b = c(159, 123, 131, 141),
  effect = c(-24.0, -31.2, 27.5, -5.9),
  ci_low = c(-42.1, -54.0, 4.7, -27.7),
  ci_high = c(-6.8, -8.8, 52.0, 15.8),
  p_value = c(0.006, 0.007, 0.018, 0.607),
  adoption = c(0.679, 0.394, 0.379, 0.297),
  anomaly = c(FALSE, FALSE, TRUE, FALSE),
  stringsAsFactors = FALSE
)

weekly_effects$bar_color <- ifelse(weekly_effects$anomaly, pal$anomaly, pal$active)

# -- 3. Assertions -----------------------------------------------------------

# Panel A
stopifnot(nrow(df) == 21)
stopifnot(all(df$adoption_rate >= 0 & df$adoption_rate <= 1))
stopifnot(max(df$adoption_rate) > 0.78)
# Panel B
stopifnot(nrow(weekly_effects) == 4)
stopifnot(weekly_effects$effect[1] == -24.0)
stopifnot(weekly_effects$effect[3] == 27.5)
stopifnot(sum(weekly_effects$anomaly) == 1)
stopifnot(weekly_effects$anomaly[3] == TRUE)
# Study Week 1 date validation
week1_dates <- df$date[df$date >= as.Date("2025-11-09") & df$date <= as.Date("2025-11-16")]
stopifnot(length(unique(week1_dates)) == 6)
message("[OK] Figure 3 assertions passed")

# -- 4. Panel A -- Daily adoption rate ---------------------------------------

date_range <- as.Date(c("2025-11-08", "2025-12-08"))
week_boundaries <- as.Date(c("2025-11-16", "2025-11-23", "2025-11-30"))
week_label_dates <- as.Date(c("2025-11-12", "2025-11-19", "2025-11-26", "2025-12-04"))

p_adoption <- ggplot(df, aes(x = date, y = adoption_rate)) +
  geom_vline(xintercept = week_boundaries, linetype = "dashed",
    color = "gray70", linewidth = 0.4) +
  annotate("text", x = week_label_dates, y = 0.97,
    label = c("Week 1", "Week 2", "Week 3", "Week 4"),
    size = 3.2, color = "gray50", family = "Arial") +
  geom_hline(yintercept = 0.443, linetype = "dashed", color = "gray50",
    linewidth = 0.5) +
  geom_smooth(method = "glm", method.args = list(family = "quasibinomial"),
    aes(weight = n_patients),
    color = pal$emphasis, fill = pal$ci_band, alpha = 0.3,
    linewidth = 0.8, se = TRUE) +
  geom_point(color = pal$active, alpha = 0.8, size = 2.5) +
  scale_y_continuous(labels = scales::percent_format(),
    limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  scale_x_date(date_breaks = "5 days", date_labels = "%b %d",
    limits = date_range) +
  annotate("text", x = as.Date("2025-12-08"), y = 0.47,
    label = "Overall: 44.3%", hjust = 1, size = 2.8,
    color = "gray50", family = "Arial") +
  labs(x = NULL, y = "Daily Adoption Rate") +
  theme_shaked(base_size = 11) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray92", linewidth = 0.3),
    plot.margin = margin(5, 5, 2, 5)
  )

# -- 5. Panel B -- Weekly ITT effect bars (categorical x-axis) ---------------

weekly_effects$effect_label <- sprintf("%.1f min", weekly_effects$effect)
weekly_effects$adoption_label <- sprintf("%d%%", round(weekly_effects$adoption * 100))

p_weekly <- ggplot(weekly_effects, aes(x = week_label, y = effect)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black",
    linewidth = 0.5) +
  geom_col(aes(fill = bar_color), width = 0.7, alpha = 0.85) +
  scale_fill_identity() +
  scale_y_continuous(breaks = seq(-40, 40, 20), limits = c(-45, 45)) +
  labs(x = NULL, y = "ITT Effect (minutes, HL)") +
  theme_shaked(base_size = 11) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray92", linewidth = 0.3),
    plot.margin = margin(2, 5, 5, 5)
  )

# -- 6. Assembly -------------------------------------------------------------

fig3 <- (p_adoption / p_weekly) +
  plot_layout(heights = c(0.6, 0.4)) +
  plot_annotation(
    tag_levels = "A",
    tag_prefix = "", tag_suffix = "",
    theme = theme(plot.tag = element_text(face = "bold", size = 14))
  )

# -- 7. Save -----------------------------------------------------------------

save_figure(fig3, "Figure3_Adoption_Dynamics", width = 7, height = 6)
cat("[DONE] Figure 3 complete.\n")
