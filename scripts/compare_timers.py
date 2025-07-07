import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("results_all2.csv")

# Filtra solo i CSR_apd e CSR_spd (qualsiasi implementazione)
df_csr = df[df["type"].str.contains("CSR_apd|CSR_apr")].copy()

# Estrai la modalità (apd o spd)
df_csr["mode"] = df_csr["type"].apply(lambda x: x.split('_')[1])

# Calcola medie per ogni componente, separando per modalità
avg_components = df_csr.groupby("mode")[["comm_dist", "comm_aggr", "computation_sort", "computation_mult"]].mean()

# Dati per il grafico
modes = ["apd", "apr"]
labels = ["CSR Direct", "CSR Reverse"]
colors = {
    "comm_dist": "deepskyblue",
    "comm_aggr": "limegreen",
    "computation_sort": "gold",
    "computation_mult": "red"
}

custom_labels = {
    "comm_dist": "Coms. Distribution Time",
    "computation_sort": "Comp. Sorting Time",
    "computation_mult": "Comp. Multiplication Time",
    "comm_aggr": "Coms. Aggregation Time"
}

# Ordine componenti da visualizzare
components = ["comm_dist", "computation_sort", "computation_mult", "comm_aggr"]

# Inizializza grafico
fig, ax = plt.subplots(figsize=(5, 5))
bar_width = 0.5
x = np.arange(len(modes))
bottom = np.zeros(len(modes))

# Stack di ogni componente
for comp in components:
    heights = avg_components[comp].reindex(modes).values
    ax.bar(x, heights, bar_width, bottom=bottom, label=custom_labels[comp], color=colors[comp])
    bottom += heights

# Etichette
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Average total time (s)")
ax.set_title("Average CSR script time")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

plt.tight_layout()
plt.show()
