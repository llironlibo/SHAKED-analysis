# =============================================================================
# efigure1_propensity_balance.R
# SHAKED appendix eFigure 1: Propensity-score distribution + SMD love plot.
#
# Panel A: PS density overlap (near-random wing allocation)
# Panel B: Horizontal love plot (pre- vs post-weighting SMD)
#
# Reproduces: eFigure 1 (propensity-score distribution).
# Input : data/cleaned/unified_pilot_cohort.csv,
#         results/eFigure4_IPTW_Balance_data.csv  (written by stage 05)
# Output: results/figures/eFigure1_Propensity_Balance_600dpi.png + .pdf
# Size: 7 x 6 inches
# =============================================================================

# Resolve _common.R next to this script (portable across CWDs).
.this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
source(file.path(dirname(.this_file), "_common.R"))

suppressPackageStartupMessages({
  library(pROC)
})

# -- 1. Load data ------------------------------------------------------------

df <- read_csv(file.path(DATA_DIR, "unified_pilot_cohort.csv"),
  show_col_types = FALSE)
df <- df |> filter(n_consultations > 0)

stopifnot(nrow(df) == 1138)

cols <- names(df)
df$wing_a <- as.integer(df$wing == "A")
df$age <- as.numeric(df[[cols[3]]])
# is_female: the sex field stores a categorical label; match the "female" value.
# "female" is a neutral placeholder for the controlled-access raw label.
df$is_female <- as.integer(grepl("female", df[[cols[4]]]))
df$triage_p1 <- as.integer(df$triage_acuity_numeric == 1)
df$triage_p2 <- as.integer(df$triage_acuity_numeric == 2)
df$triage_p3 <- as.integer(df$triage_acuity_numeric == 3)
df$radiology <- as.integer(df$has_radiology_consult)
df$admitted <- as.integer(df$is_admitted)

# -- 2. Propensity-score model -----------------------------------------------

ps_model <- glm(
  wing_a ~ age + is_female + triage_p1 + triage_p2 + triage_p3 +
    radiology + admitted + diagnosis_count,
  family = binomial, data = df, na.action = na.exclude
)
df$ps <- predict(ps_model, type = "response")

auc_val <- as.numeric(auc(roc(df$wing_a, df$ps, quiet = TRUE)))
stopifnot(abs(auc_val - 0.543) < 0.02)
message(sprintf("[OK] PS AUC = %.3f (expected ~0.543)", auc_val))

# -- 3. Panel A -- PS distribution overlap -----------------------------------

p_ps <- ggplot(df, aes(x = ps, fill = wing, color = wing)) +
  geom_density(alpha = 0.35, linewidth = 0.6) +
  scale_fill_manual(
    values = c("A" = pal$active, "B" = pal$control),
    labels = c("A" = "Wing A", "B" = "Wing B")) +
  scale_color_manual(
    values = c("A" = pal$active, "B" = pal$control),
    labels = c("A" = "Wing A", "B" = "Wing B")) +
  annotate("label",
    x = 0.7, y = Inf, vjust = 1.5,
    label = sprintf("AUC = %.3f\n(near-random allocation)", auc_val),
    size = 3.5, family = "Arial", fill = "white",
    linewidth = 0.3, label.padding = unit(0.3, "lines")) +
  labs(x = "Propensity Score P(Wing A | covariates)", y = "Density") +
  theme_shaked(base_size = 11) +
  theme(
    legend.position = c(0.15, 0.85),
    legend.title = element_blank(),
    legend.text = element_text(size = 11),
    legend.background = element_rect(fill = "white", color = NA),
    panel.grid.major.y = element_line(color = "gray92", linewidth = 0.3)
  )

# -- 4. Panel B -- Horizontal love plot --------------------------------------

smd_data <- read_csv(IPTW_CSV, show_col_types = FALSE)
stopifnot(nrow(smd_data) > 0)
stopifnot(all(c("SMD_Pre", "SMD_Post") %in% names(smd_data)))

smd_a <- smd_data |> filter(grepl("Model A", Model))

cov_labels <- c(
  "age" = "Age", "is_female" = "Female", "triage" = "Triage Acuity",
  "diagnosis_count" = "Diagnosis Count", "has_radiology" = "Radiology Consult",
  "is_admitted" = "Admitted", "n_consultations" = "N Consultations",
  "is_ambulance" = "Ambulance Arrival", "admission_hour" = "Admission Hour"
)
smd_a$cov_label <- cov_labels[smd_a$Covariate]
smd_a$cov_label[is.na(smd_a$cov_label)] <- smd_a$Covariate[is.na(smd_a$cov_label)]

smd_a <- smd_a |> arrange(abs(SMD_Pre))
smd_a$cov_label <- factor(smd_a$cov_label, levels = smd_a$cov_label)

p_smd <- ggplot(smd_a) +
  geom_hline(yintercept = c(-0.1, 0.1), linetype = "dashed",
    color = pal$anomaly, linewidth = 0.5) +
  geom_hline(yintercept = 0, color = "gray50", linewidth = 0.3) +
  geom_segment(aes(x = cov_label, xend = cov_label,
    y = SMD_Pre, yend = SMD_Post),
    arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
    color = "gray60", linewidth = 0.4) +
  geom_point(aes(x = cov_label, y = SMD_Pre),
    shape = 1, size = 3.5, color = pal$control) +
  geom_point(aes(x = cov_label, y = SMD_Post),
    shape = 16, size = 3.5, color = pal$active) +
  annotate("point", x = 1, y = 0.42, shape = 1, size = 3, color = pal$control) +
  annotate("text", x = 1, y = 0.42, label = "  Pre-weighting",
    hjust = 0, size = 3.2, family = "Arial") +
  annotate("point", x = 1, y = 0.35, shape = 16, size = 3, color = pal$active) +
  annotate("text", x = 1, y = 0.35, label = "  Post-weighting (IPTW)",
    hjust = 0, size = 3.2, family = "Arial") +
  scale_y_continuous(
    breaks = c(-0.25, -0.1, 0, 0.1, 0.25),
    limits = c(-0.35, 0.5)) +
  labs(x = NULL, y = "Standardized Mean Difference") +
  theme_shaked(base_size = 11) +
  theme(
    panel.grid.major.y = element_line(color = "gray92", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10)
  )

# -- 5. Assembly -------------------------------------------------------------

fig <- (p_ps / p_smd) +
  plot_layout(heights = c(0.45, 0.55)) +
  plot_annotation(
    tag_levels = "A",
    tag_prefix = "", tag_suffix = "",
    theme = theme(plot.tag = element_text(face = "bold", size = 14))
  )

# -- 6. Save -----------------------------------------------------------------

save_figure(fig, "eFigure1_Propensity_Balance", width = 7, height = 6)
message("[OK] eFigure 1 assertions passed")
cat("[DONE] eFigure 1 complete.\n")
