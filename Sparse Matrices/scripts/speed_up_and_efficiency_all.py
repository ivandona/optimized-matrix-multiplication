import pandas as pd

# === CONFIG ===
INPUT_CSV = "results_CSR_reverse_pack.csv"

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)
df["cores"] = df["processes"] * df["threads"]
df["total_time"] = df["total_time"] - df["generation_time"]

# Calcola statistiche: mean, min, max
agg = df.groupby(["type", "processes", "threads", "cores"]).agg({
    "total_time": ["mean", "min", "max"]
}).reset_index()
agg.columns = ["type", "processes", "threads", "cores", "mean", "min", "max"]

# Estrai tempi seriali (usati per i tre gruppi)
serial_row = agg[agg["type"] == "CSR_a_seq"].iloc[0]
t_serials = {
    "min": serial_row["min"],
    "mean": serial_row["mean"],
    "max": serial_row["max"]
}

# Filtra paralleli
agg = agg[agg["type"] != "CSR_a_seq"].copy()

# Calcola speedup ed efficienza rispetto ai rispettivi t_serial
for metric in ["min", "mean", "max"]:
    agg[f"speedup_{metric}"] = t_serials[metric] / agg[metric]
    agg[f"efficiency_{metric}"] = agg[f"speedup_{metric}"] / agg["cores"]

# Aggiungi modalità
agg["mode"] = agg["type"].apply(lambda x: x.split('_')[-1])

# === Funzione per creare tabella ===
def make_compact_table(df, metric_base, include_one_core=True, label=""):
    rows = []

    # OpenMP
    df_omp = df[(df["mode"] == "OpenMP") & (df["processes"] == 1)]
    row = {}
    for _, r in df_omp.iterrows():
        val = f"{r[f'{metric_base}_min']:.2f} {r[f'{metric_base}_mean']:.2f} {r[f'{metric_base}_max']:.2f}"
        row[int(r["cores"])] = val
    if include_one_core:
        row[1] = "1.00 1.00 1.00"
    rows.append(pd.Series(row, name="CSR_apr_OpenMP"))

    # MPI
    df_mpi = df[(df["mode"] == "MPI") & (df["threads"] == 1)]
    row = {}
    for _, r in df_mpi.iterrows():
        val = f"{r[f'{metric_base}_min']:.2f} {r[f'{metric_base}_mean']:.2f} {r[f'{metric_base}_max']:.2f}"
        row[int(r["cores"])] = val
    if include_one_core:
        row[1] = "1.00 1.00 1.00"
    rows.append(pd.Series(row, name="CSR_apr_MPI"))

    # Hybrid
    df_hybrid = df[df["mode"] == "Hybrid"]
    for proc in sorted(df_hybrid["processes"].unique()):
        row = {}
        df_proc = df_hybrid[df_hybrid["processes"] == proc]
        for _, r in df_proc.iterrows():
            core_count = int(r["cores"])
            val = f"{r[f'{metric_base}_min']:.2f} {r[f'{metric_base}_mean']:.2f} {r[f'{metric_base}_max']:.2f}"
            row[core_count] = val
        if include_one_core:
            row[1] = "1.00 1.00 1.00"
        rows.append(pd.Series(row, name=f"CSR_apr_Hybrid ({proc} pr)"))

    # Combina e ordina
    result = pd.DataFrame(rows)
    result = result[sorted(result.columns)]
    return result.fillna("-")

# === Costruisci tabelle e stampale ===
speedup_tbl = make_compact_table(agg, "speedup")
efficiency_tbl = make_compact_table(agg, "efficiency")

print("\n=== SPEEDUP (min, medio, max) ===")
print(speedup_tbl.to_string())

print("\n=== EFFICIENZA (min, medio, max) ===")
print(efficiency_tbl.to_string())
