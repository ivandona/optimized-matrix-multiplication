import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carica CSV
df = pd.read_csv("results_reverse.csv")

# Aggiungi colonne helper
df["pt_combo"] = df["processes"].astype(str) + "pr-" + df["threads"].astype(str) + "th"
df["mode_label"] = df["type"].apply(lambda x: "_".join(x.split("_")[:2]))  # es. COO_apr
df["total_time"] = df["total_time"] - df["generation_time"]

# Raggruppa per media
df_avg = df.groupby(["mode_label", "nodes", "processes", "threads", "pt_combo"]).mean(numeric_only=True).reset_index()
min_time = df_avg["total_time"].min()

# Ordina pt_combo globalmente
df_avg["pt_order"] = list(zip(df_avg["processes"], df_avg["threads"]))
pt_combos_sorted = df_avg.sort_values(by=["processes", "threads"])["pt_combo"].unique()

# Colori fissi per combinazioni pt_combo
cmap = plt.get_cmap("tab10")
pt_colors = {pt: cmap(i % 10) for i, pt in enumerate(pt_combos_sorted)}

# Nodi ordinati
node_values = sorted(df_avg["nodes"].unique())
x_pos = np.arange(len(node_values))
bar_width = 0.8 / max(len(pt_combos_sorted), 1)

# Modalità da graficare
modes = sorted(df_avg["mode_label"].unique())
modes[0], modes[1] = modes[1], modes[0]

# Crea 3 subplot impilati
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)
fig.suptitle("Average total time of Reverse mode (256 cores)", fontsize=14, fontweight='bold')

ymax = df_avg["total_time"].max()

for ax, mode in zip(axes, modes):
    df_mode = df_avg[df_avg["mode_label"] == mode]
    ax.axhline(min_time, linestyle="--", color="black", linewidth=1)
    for idx, node in enumerate(node_values):
        df_node = df_mode[df_mode["nodes"] == node]
        pt_combos_node = df_node.sort_values(by=["processes", "threads"])["pt_combo"].unique()
        num_bars = len(pt_combos_node)
        # centro dinamico
        start_offset = - (num_bars - 1) / 2 * bar_width

        for i, pt in enumerate(pt_combos_node):
            row = df_node[df_node["pt_combo"] == pt]
            if not row.empty:
                time = row["total_time"].values[0]
                offset = start_offset + i * bar_width
                ax.bar(x_pos[idx] + offset, time, width=bar_width, color=pt_colors[pt])
    mode_correct = mode.replace("_", r"\_")
    ax.set_ylabel(f"Total time (s)\n$\\bf{{{mode_correct}}}$")
    ax.set_ylim(0, ymax+5)
    ax.grid(True, linestyle='--', alpha=0.4)

# Asse X (solo ultimo grafico)
axes[-1].set_xticks(x_pos)
axes[-1].set_xticklabels([str(n) for n in node_values])
axes[-1].set_xlabel("Number of scattered nodes")

# Legenda unica fuori dai subplot
handles = [plt.Line2D([0], [0], marker='s', color='w', label=pt,
                      markerfacecolor=pt_colors[pt], markersize=8)
           for pt in pt_combos_sorted]
fig.legend(
    handles=handles,
    title="Processes-Threads",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),  # <--- abbassa leggermente la legenda
    ncol=5,
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 1, 0.94])  # lascia spazio per la legenda sopra
plt.show()
