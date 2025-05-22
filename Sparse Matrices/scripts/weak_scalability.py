import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_PACK = "results_CSR_reverse_pack_weak_scalability.csv"
CSV_SCATTER = "results_CSR_reverse_scatter_weak_scalability.csv"

def load_and_process(csv_file, label):
    df = pd.read_csv(csv_file)
    df["cores"] = df["processes"] * df["threads"]
    df["total_time"] = df["total_time"] - df["generation_time"]

    # Considera anche la dimensione del problema (assumiamo sia la stessa per tutti)
    if "n" not in df.columns:
        raise ValueError("Il CSV deve contenere una colonna 'n' per la dimensione del problema")

    agg = df.groupby(["type", "processes", "threads", "cores", "n"]).agg({
        "total_time": ["mean", "min", "max"]
    }).reset_index()
    agg.columns = ["type", "processes", "threads", "cores", "n", "mean", "min", "max"]

    # Aggiungi modalità e origine
    agg["mode"] = agg["type"].apply(lambda x: x.split('_')[-1])
    agg["origin"] = label
    return agg

# === Caricamento dati pack e scatter ===
agg_pack = load_and_process(CSV_PACK, "pack:excl")
agg_scatter = load_and_process(CSV_SCATTER, "scatter")

# === Unione dati ===
agg_all = pd.concat([agg_pack, agg_scatter], ignore_index=True)

# === Funzione tabella compatta ===
def make_compact_table(df, metric):
    rows = []
    x_labels = []

    # OpenMP solo per 'pack'
    df_omp = df[(df["mode"] == "OpenMP") & (df["origin"] == "pack:excl") & (df["processes"] == 1)]
    row = {}
    for _, r in df_omp.iterrows():
        key = f"{int(r['cores'])}; {int(r['n'])}"
        row[key] = f"{r[metric]:.2f}"
        x_labels.append(key)
    rows.append(pd.Series(row, name="OpenMP [pack:excl]"))

    # MPI
    for origin in ["pack:excl", "scatter"]:
        df_mpi = df[(df["mode"] == "MPI") & (df["origin"] == origin) & (df["threads"] == 1)]
        row = {}
        for _, r in df_mpi.iterrows():
            key = f"{int(r['cores'])}; {int(r['n'])}"
            row[key] = f"{r[metric]:.2f}"
            x_labels.append(key)
        rows.append(pd.Series(row, name=f"MPI [{origin}]"))

    # Hybrid
    for origin in ["pack:excl", "scatter"]:
        df_hybrid = df[(df["mode"] == "Hybrid") & (df["origin"] == origin)]
        for proc in sorted(df_hybrid["processes"].unique()):
            row = {}
            df_proc = df_hybrid[df_hybrid["processes"] == proc]
            for _, r in df_proc.iterrows():
                key = f"{int(r['cores'])}; {int(r['n'])}"
                row[key] = f"{r[metric]:.2f}"
                x_labels.append(key)
            rows.append(pd.Series(row, name=f"Hybrid [{origin}] ({proc} pr)"))

    result = pd.DataFrame(rows)
    return result.fillna("-"), sorted(set(x_labels), key=lambda x: int(x.split(';')[0]))

# === Tabella tempi ===
time_tbl, x_labels = make_compact_table(agg_all, "mean")

# === Estrazione numerica ===
time_numeric = time_tbl.copy()
time_numeric = time_numeric.map(lambda x: float(x) if x != "-" else None)

# === Grafico ===
plt.figure(figsize=(10, 5))

# === Scala reale per l'asse x
x_real = [int(x.split(';')[0]) for x in x_labels]  # es. [1, 2, 4, 8, 16, 32, 64]

# Sostituisci tutte le plt.plot(x_labels, ...) con plt.plot(x_real, ..., label=..., ...)
plt.plot(x_real,
         time_numeric.loc["OpenMP [pack:excl]"][x_labels],
         label="OpenMP [pack:excl]", marker="o", color="limegreen")

# MPI
for origin, marker, color in zip(["pack:excl", "scatter"], ["s", "D"], ["tab:orange", "red"]):
    plt.plot(x_real,
             time_numeric.loc[f"MPI [{origin}]"][x_labels],
             label=f"MPI [{origin}]", marker=marker, color=color)

# Hybrid best
for origin, marker, color in zip(["pack:excl", "scatter"], ["^", "v"], ["deepskyblue", "slateblue"]):
    hybrid_rows = time_numeric[time_numeric.index.str.startswith(f"Hybrid [{origin}]")]
    hybrid_best = hybrid_rows.min()
    plt.plot(x_real,
             hybrid_best[x_labels],
             label=f"Hybrid [{origin}] (best per cores)", marker=marker, color=color)

# === Linea ideale
ideal_time = time_numeric.min().min()
plt.hlines(y=ideal_time, xmin=min(x_real), xmax=max(x_real),
           linestyle="--", color="black", label="Ideal")


# === Layout aggiornato
plt.xlabel("Cores; Matrix Side Size")
plt.ylabel("Total Time (s)")
plt.title("Weak Scalability - Time vs Cores and Problem Size (d=0.15)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.xticks(x_real, x_labels, rotation=45)  # Tick con etichette testuali
plt.tight_layout()
plt.show()
