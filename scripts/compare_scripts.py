import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# load CSV
df = pd.read_csv("results_all.csv")
df["total_time"] = df["total_time"] - df["generation_time"]

# Filter sequential
df_parallel = df[~df["type"].str.contains("seq")].copy()
df_seq = df[df["type"].str.contains("seq")]

# Extract matrices info
n = df["n"].iloc[0]
density = df["density"].iloc[0]
cores = df["processes"].iloc[0] * df["threads"].iloc[0]

# Create a column for the group (processes, threads)
df_parallel["group"] = list(zip(df_parallel["processes"], df_parallel["threads"]))

# Group up by script (type, processes, threads), compute average
df_avg = df_parallel.groupby(["type", "processes", "threads"]).mean(numeric_only=True).reset_index()

# Add group column also here
df_avg["group"] = list(zip(df_avg["processes"], df_avg["threads"]))

# Extract scripts info
df_avg["format"] = df_avg["type"].apply(lambda x: x.split('_')[0])
df_avg["mode"] = df_avg["type"].apply(lambda x: x.split('_')[1])
df_avg["impl"] = df_avg["type"].apply(lambda x: x.split('_')[2])
df_avg["combo"] = df_avg["format"] + "_" + df_avg["mode"]

# Find groups (couple (p,t)) sorted
groups = sorted(df_avg["group"].unique(), key=lambda x: (x[0], x[1]))
x = np.arange(len(groups))
ymax = df_avg["total_time"].max()

# Shape for implementation
impl_shape = {
    "OpenMP": "o",
    "Hybrid": "^",
    "MPI": "D"
}

# Colors for implementation
impl_colors = {
    "OpenMP": "gold",
    "Hybrid": "tab:orange",
    "MPI": "red"
}

# Colors for the 6 combinations
combo_list = df_avg["combo"].unique()
combo_colors = {
    "COO_spr": "limegreen",
    "COO_apr": "darkgreen",
    "CSR_apr": "tab:olive",
    "COO_spd": "deepskyblue",
    "COO_apd": "darkblue",
    "CSR_apd": "slateblue"
}

# Init plot
fig, ax = plt.subplots(figsize=(7, 5))

# Draw points e harvest data for lines
for impl in impl_colors:
    df_impl = df_avg[df_avg["impl"] == impl]
    for _, row in df_impl.iterrows():
        xi = groups.index(row["group"])
        ax.scatter(xi, row["total_time"], marker=impl_shape[impl], color=impl_colors[impl], s=50)

# draw the lines
for combo in combo_list:
    df_combo = df_avg[df_avg["combo"] == combo]
    df_combo_sorted = df_combo.sort_values(by=["processes", "threads"])
    xs = [groups.index(g) for g in df_combo_sorted["group"]]
    ys = df_combo_sorted["total_time"].values
    ax.plot(xs, ys, '-', color=combo_colors[combo], label=combo)

# Axis X
ax.set_xticks(x)
ax.set_xticklabels([f"{g[0]}pr-{g[1]}th" for g in groups], rotation=45)
ax.set_xlabel("Resources allocation (processes-threads)")
ax.set_ylabel("Total time (s)")
ax.set_yscale("linear")
ax.set_ylim(0, ymax+100)
ax.set_title(f"Average sparse matrix multiplication time\n (dimension={n}, density={density}, total cores={cores})")

# Custom legend
point_legend = [Line2D([0], [0], marker=impl_shape[impl], color='w', label=impl,
                       markerfacecolor=color, markersize=8)
                for impl, color in impl_colors.items()]
line_legend = [Line2D([0], [0], color=color, lw=2, label=combo)
               for combo, color in combo_colors.items()]

# Combine the two legends
ax.legend(handles=point_legend + line_legend, loc='upper right', ncol=2, fontsize=9)

plt.tight_layout()
plt.show()

# Print average sequential script time
if not df_seq.empty:
    print("\nAverage sequential script time:")
    df_seq_avg = df_seq.groupby("type").mean(numeric_only=True).reset_index()
    for _, row in df_seq_avg.iterrows():
        print(f"{row['type']}: {row['total_time']:.2f} s")
