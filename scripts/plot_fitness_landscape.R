#!/usr/bin/env Rscript
# ggplot2 dashboard for fitness landscape JSON exports.
# Usage: Rscript scripts/plot_fitness_landscape.R [path/to/fitness_landscape.json]

args <- commandArgs(trailingOnly = TRUE)
json_path <- if (length(args) >= 1) args[[1]] else "output/fitness_landscape_medium/fitness_landscape.json"
out_dir <- dirname(json_path)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Install jsonlite: install.packages('jsonlite')")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2: install.packages('ggplot2')")
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

data <- fromJSON(json_path, simplifyVector = FALSE)
cells <- as.data.frame(do.call(rbind, lapply(data$cells, as.data.frame, stringsAsFactors = FALSE)))

cells$architecture <- factor(cells$architecture)
cells$founder_mode <- factor(cells$founder_mode)

p_heat <- ggplot(cells, aes(x = x_label, y = y_label, fill = fitness_mean)) +
  geom_tile(color = "white", linewidth = 0.3) +
  facet_grid(founder_mode ~ architecture, scales = "free") +
  scale_fill_viridis_c(option = "C", na.value = "grey85") +
  labs(
    title = sprintf("Fitness landscape (%s steps)", data$steps),
    subtitle = sprintf("build %s · %s replicates", data$build_id, data$replicates),
    fill = "Fitness",
    x = "X axis",
    y = "Y axis"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
    axis.text.y = element_text(size = 7),
    panel.grid = element_blank()
  )

ggsave(file.path(out_dir, "fitness_landscape_ggplot.png"), p_heat, width = 14, height = 9, dpi = 150)
cat("Wrote:", file.path(out_dir, "fitness_landscape_ggplot.png"), "\n")

runs <- as.data.frame(do.call(rbind, lapply(data$runs, function(r) {
  if (length(r$trajectory) == 0) return(NULL)
  traj <- as.data.frame(do.call(rbind, lapply(r$trajectory, as.data.frame, stringsAsFactors = FALSE)))
  cbind(
    data.frame(
      architecture = r$architecture,
      x_label = r$x_label,
      y_label = r$y_label,
      founder_mode = r$founder_mode,
      seed = r$seed,
      stringsAsFactors = FALSE
    ),
    traj
  )
})))

if (nrow(runs) > 0) {
  # Pick top fitness cell per architecture/founder for trajectory panel
  top_cells <- aggregate(fitness_score ~ architecture + founder_mode + x_label + y_label, runs, max)
  top_cells <- top_cells[order(-top_cells$fitness_score), ]
  top_cells <- top_cells[!duplicated(top_cells[, c("architecture", "founder_mode")]), ]

  sel <- merge(
    runs,
    top_cells[, c("architecture", "founder_mode", "x_label", "y_label")],
    by = c("architecture", "founder_mode", "x_label", "y_label")
  )
  sel$label <- paste(sel$architecture, sel$x_label, sel$y_label, sep = " · ")

  p_time <- ggplot(sel, aes(x = step, y = fitness, color = factor(seed), group = seed)) +
    geom_line(alpha = 0.85, linewidth = 0.7) +
    facet_grid(founder_mode ~ architecture, scales = "free_y") +
    labs(
      title = "Fitness through time (best cell per architecture)",
      x = "Step",
      y = "Fitness score",
      color = "Seed"
    ) +
    theme_minimal(base_size = 11)

  ggsave(file.path(out_dir, "fitness_landscape_trajectories_ggplot.png"), p_time, width = 14, height = 8, dpi = 150)
  cat("Wrote:", file.path(out_dir, "fitness_landscape_trajectories_ggplot.png"), "\n")
}
