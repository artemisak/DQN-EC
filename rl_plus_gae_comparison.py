import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Load CSVs
# -------------------------
methods = {
    "Double DQN + Communication": pd.read_csv("./runs/run_20250714_155357/Double DQN + Communication.csv"),
    "Double DQN": pd.read_csv("./runs/run_20250714_155357/Double DQN.csv"),
    "DQN": pd.read_csv("./runs/run_20250714_155357/DQN.csv"),
}

# -------------------------
# Metrics of interest
# -------------------------
metrics = [
    "listener_loss",
    "speaker_loss",
    "average_reward_last_10"
]

metric_mapping = {
    "average_reward_last_10": "Average Reward (Last 10)",
    "speaker_loss": "Speaker Loss",
    "listener_loss": "Listener Loss"
}

# Colors
colors = {
    "Double DQN + Communication": "blue",
    "Double DQN": "red",
    "DQN": "green",
}

SMOOTH_WINDOW = 10

def smooth_series(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1).mean()

# -------------------------
# Custom x-axis formatter
# -------------------------
def custom_comma_formatter(x, pos):
    if abs(x) >= 10000:
        return f"{int(x):,}"
    else:
        return f"{int(x)}"

# -------------------------
# Sort DataFrames
# -------------------------
for name in methods:
    methods[name] = methods[name].sort_values("global_step").reset_index(drop=True)

# -------------------------
# Plotting
# -------------------------
with PdfPages("./rl_metrics-no-title.pdf") as pdf:
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6), sharex=False)
    if len(metrics) == 1:
        axes = [axes]

    subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, metric, label in zip(axes, metrics, subplot_labels):
        # Collect all data points for legend placement
        all_xs, all_ys = [], []

        for name, df in methods.items():
            df_plot = df
            grouped = df_plot.groupby("global_step")[metric].agg(["mean","std"]).dropna()
            if grouped.empty:
                continue
            grouped["mean"] = smooth_series(grouped["mean"])
            grouped["std"] = smooth_series(grouped["std"])

            x = grouped.index.values
            y = grouped["mean"].values

            all_xs.append(x)
            all_ys.append(y)

            ax.plot(x, y, label=name, color=colors[name])
            ax.fill_between(x, y-grouped["std"].values, y+grouped["std"].values,
                            color=colors[name], alpha=0.2)

        # Flatten arrays
        if all_xs:
            all_xs = np.concatenate(all_xs)
            all_ys = np.concatenate(all_ys)
        else:
            all_xs = np.array([])
            all_ys = np.array([])

        # Axis labels and grid
        ax.set_xlabel("Global Step")
        ax.set_ylabel(metric_mapping[metric])
        ax.grid(True)
        ax.xaxis.set_major_formatter(FuncFormatter(custom_comma_formatter))

        ax.legend(fontsize=8, frameon=True)

        # Subplot label under plot
        ax.text(0.5, -0.32, label, transform=ax.transAxes,
                fontsize=12, ha='center', va='top')

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    pdf.savefig(fig, dpi=800, bbox_inches="tight")
    plt.close(fig)

print("✅ Saved PDF: results/all_metrics_comparison_one_row.pdf (titles above, legends inside, labels below)")
