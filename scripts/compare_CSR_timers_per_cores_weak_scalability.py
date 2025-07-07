import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("results_CSR_reverse_pack_weak_scalability.csv")

# Sottrai il tempo di generazione
df["total_time"] = df["total_time"] - df["generation_time"]

# Filtra solo Hybrid
df_hybrid = df[df["type"].str.contains("Hybrid")].copy()

# Calcola numero di cores
df_hybrid["cores"] = df_hybrid["processes"] * df_hybrid["threads"]

# Raggruppa per (processes, threads) e calcola la media dei tempi
grouped = df_hybrid.groupby(["processes", "threads"]).mean(numeric_only=True).reset_index()

# Ricalcola i cores dopo il groupby
grouped["cores"] = grouped["processes"] * grouped["threads"]

# Per ogni numero di core, prendi la riga con total_time minimo
best_per_core = grouped.loc[grouped.groupby("cores")["total_time"].idxmin()]
best_per_core = best_per_core.sort_values("cores")

# Componenti da usare
components = ["comm_dist", "computation_sort", "computation_mult", "comm_aggr"]

best_components = best_per_core[components]

# Dati per il grafico
x = np.arange(len(best_per_core))
bar_width = 0.6

colors = {
    "comm_dist": "deepskyblue",
    "comm_aggr": "limegreen",
    "computation_sort": "gold",
    "computation_mult": "red"
}

custom_labels = {
    "comm_dist": "Coms. Distribution Time",
    "comm_aggr": "Coms. Aggregation Time",
    "computation_sort": "Comp. Sorting Time",
    "computation_mult": "Comp. Multiplication Time"
}

# Inizializza grafico
fig, ax = plt.subplots(figsize=(8, 6))
bottom = np.zeros(len(best_components))

# Disegna barre impilate
for comp in components:
    heights = best_components[comp].values
    ax.bar(x, heights, bar_width, bottom=bottom, label=custom_labels[comp], color=colors[comp])
    bottom += heights

# Etichette asse x
ax.set_xticks(x)
ax.set_xticklabels(best_per_core["cores"])
ax.set_xlabel("Number of Cores (processes × threads)")
ax.set_ylabel("Normalized Time (s)")
ax.set_title("CSR Reverse Hybrid - Best Config per Cores (Normalized per-core workload)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
