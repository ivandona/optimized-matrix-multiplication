import pandas as pd

# === CONFIG ===
INPUT_CSV = "results_CSR_reverse_scatter.csv" #  pack/scatter

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)
df["cores"] = df["processes"] * df["threads"]
df["total_time"] = df["total_time"] - df["generation_time"]

# Calcola statistiche: mean, min, max
agg = df.groupby(["type", "processes", "threads", "cores"]).agg({
    "total_time": ["mean", "min", "max"]
}).reset_index()
agg.columns = ["type", "processes", "threads", "cores", "mean", "min", "max"]

# Estrai tempo seriale medio
serial_row = agg[agg["type"] == "CSR_a_seq"].iloc[0]
t_serial_mean = serial_row["mean"]

# Filtra paralleli
agg = agg[agg["type"] != "CSR_a_seq"].copy()

# Calcola speedup e efficienza basati solo sulla media
agg["speedup"] = t_serial_mean / agg["mean"]
agg["efficiency"] = agg["speedup"] / agg["cores"]

# Aggiungi modalità
agg["mode"] = agg["type"].apply(lambda x: x.split('_')[-1])

# === Funzione per creare tabella compatta ===
def make_compact_table(df, metric, include_one_core=True, label=""):
    rows = []

    # OpenMP
    df_omp = df[(df["mode"] == "OpenMP") & (df["processes"] == 1)]
    row = {}
    for _, r in df_omp.iterrows():
        row[int(r["cores"])] = f"{r[metric]:.2f}"
    if include_one_core:
        row[1] = "1.00"
    rows.append(pd.Series(row, name="CSR_apr_OpenMP"))

    # MPI
    df_mpi = df[(df["mode"] == "MPI") & (df["threads"] == 1)]
    row = {}
    for _, r in df_mpi.iterrows():
        row[int(r["cores"])] = f"{r[metric]:.2f}"
    if include_one_core:
        row[1] = "1.00"
    rows.append(pd.Series(row, name="CSR_apr_MPI"))

    # Hybrid
    df_hybrid = df[df["mode"] == "Hybrid"]
    for proc in sorted(df_hybrid["processes"].unique()):
        row = {}
        df_proc = df_hybrid[df_hybrid["processes"] == proc]
        for _, r in df_proc.iterrows():
            row[int(r["cores"])] = f"{r[metric]:.2f}"
        if include_one_core:
            row[1] = "1.00"
        rows.append(pd.Series(row, name=f"CSR_apr_Hybrid ({proc} pr)"))

    result = pd.DataFrame(rows)
    result = result[sorted(result.columns)]
    return result.fillna("-")

# === Costruisci e stampa tabelle ===
speedup_tbl = make_compact_table(agg, "speedup")
efficiency_tbl = make_compact_table(agg, "efficiency")

print("\n=== SPEEDUP (media) ===")
print(speedup_tbl.to_string())

print("\n=== EFFICIENZA (media) ===")
print(efficiency_tbl.to_string())
