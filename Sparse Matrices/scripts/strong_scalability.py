import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_PACK = "results_CSR_reverse_pack.csv"
CSV_SCATTER = "results_CSR_reverse_scatter.csv"

def load_and_process(csv_file, label):
    df = pd.read_csv(csv_file)
    df["cores"] = df["processes"] * df["threads"]
    df["total_time"] = df["total_time"] - df["generation_time"]

    agg = df.groupby(["type", "processes", "threads", "cores"]).agg({
        "total_time": ["mean", "min", "max"]
    }).reset_index()
    agg.columns = ["type", "processes", "threads", "cores", "mean", "min", "max"]

    # Estrai tempo seriale medio
    serial_row = agg[agg["type"] == "CSR_a_seq"].iloc[0]
    t_serial_mean = serial_row["mean"]

    # Filtra paralleli
    agg = agg[agg["type"] != "CSR_a_seq"].copy()
    agg["speedup"] = t_serial_mean / agg["mean"]

    # Aggiungi modalità
    agg["mode"] = agg["type"].apply(lambda x: x.split('_')[-1])

    # Aggiungi origine (pack / scatter)
    agg["origin"] = label

    return agg

# === Caricamento dati pack e scatter ===
agg_pack = load_and_process(CSV_PACK, "pack:excl")
agg_scatter = load_and_process(CSV_SCATTER, "scatter")

# === Unisci i due dataset ===
agg_all = pd.concat([agg_pack, agg_scatter], ignore_index=True)

# === Funzione tabella compatta ===
def make_compact_table(df, metric, include_one_core=False):
    rows = []

    # OpenMP solo per 'pack'
    df_omp = df[(df["mode"] == "OpenMP") & (df["origin"] == "pack:excl") & (df["processes"] == 1)]
    row = {}
    for _, r in df_omp.iterrows():
        row[int(r["cores"])] = f"{r[metric]:.2f}"
    if include_one_core:
        row[1] = "1.00"
    rows.append(pd.Series(row, name="OpenMP [pack:excl]"))

    # MPI per entrambi
    for origin in ["pack:excl", "scatter"]:
        df_mpi = df[(df["mode"] == "MPI") & (df["origin"] == origin) & (df["threads"] == 1)]
        row = {}
        for _, r in df_mpi.iterrows():
            row[int(r["cores"])] = f"{r[metric]:.2f}"
        if include_one_core:
            row[1] = "1.00"
        rows.append(pd.Series(row, name=f"MPI [{origin}]"))

    # Hybrid per entrambi (best per core count)
    for origin in ["pack:excl", "scatter"]:
        df_hybrid = df[(df["mode"] == "Hybrid") & (df["origin"] == origin)]
        for proc in sorted(df_hybrid["processes"].unique()):
            row = {}
            df_proc = df_hybrid[df_hybrid["processes"] == proc]
            for _, r in df_proc.iterrows():
                row[int(r["cores"])] = f"{r[metric]:.2f}"
            if include_one_core:
                row[1] = "1.00"
            rows.append(pd.Series(row, name=f"Hybrid [{origin}] ({proc} pr)"))

    result = pd.DataFrame(rows)
    result = result[sorted(result.columns)]
    return result.fillna("-")

# === Tabella speedup compatta ===
speedup_tbl = make_compact_table(agg_all, "speedup")

# === Estrazione numerica ===
speedup_numeric = speedup_tbl.copy()
speedup_numeric = speedup_numeric.map(lambda x: float(x) if x != "-" else None)
core_counts = sorted([int(c) for c in speedup_numeric.columns])

# === Grafico ===
plt.figure(figsize=(8, 5))

# OpenMP solo pack
plt.plot(core_counts,
         speedup_numeric.loc["OpenMP [pack:excl]"][core_counts],
         label="OpenMP [pack:excl]", marker="o", color="limegreen")

# MPI
for origin, marker, color in zip(["pack:excl", "scatter"], ["s", "D"], ["tab:orange", "red"]):
    plt.plot(core_counts,
             speedup_numeric.loc[f"MPI [{origin}]"][core_counts],
             label=f"MPI [{origin}]", marker=marker, color=color)

# Hybrid best case per origin
for origin, marker, color in zip(["pack:excl", "scatter"], ["^", "v"], ["deepskyblue", "slateblue"]):
    hybrid_rows = speedup_numeric[speedup_numeric.index.str.startswith(f"Hybrid [{origin}]")]
    hybrid_best = hybrid_rows.max()
    plt.plot(core_counts,
             hybrid_best[core_counts],
             label=f"Hybrid [{origin}] (best per cores)", marker=marker, color=color)

# Linea ideale
plt.plot(core_counts, core_counts, label="Ideal", linestyle="--", color="black")

# Layout
plt.xlabel("Cores Numbers")
plt.ylabel("Speedup")
plt.title("Strong Scalability - Average Speedup vs Cores Numbers")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.xticks(core_counts)
plt.tight_layout()
plt.show()
