# Parallel Sparse Matrix Multiplication

This project explores **efficient parallel multiplication of large sparse matrices** using **MPI**, **OpenMP**, and **hybrid approaches**. The aim is to evaluate the performance and trade-offs of various matrix formats and distribution strategies, with a focus on scalability, memory efficiency, and computation speed.

📄 **Report**: [HPC_report_Trettel_Donà.pdf](./HPC_report_Trettel_Donà.pdf)  

---

## ✨ Key Features

- **Sparse matrix formats**: COO (Coordinate format) and CSR (Compressed Sparse Row)
- **Parallel backends**: MPI, OpenMP, and Hybrid (MPI + OpenMP)
- **Two distribution strategies**:
  - **Direct**: rows of A and the entire B are distributed per process
  - **Reverse**: distributes parts of A’s columns and B’s rows to reduce redundant computations
- **Performance benchmarking** with execution time, memory usage, speedup, and efficiency metrics

---

## ⚙️ Usage
Note: The bash scripts to compile the code and submit the jobs to a cluster work only with a PBS Altair job scheduler.

1. Clone the repository inside your cluster:
```bash
git clone https://github.com/ivandona/optimized-matrix-multiplication
cd optimized-matrix-multiplication
```

2. Run the bash script to compile the code and submit the jobs to the cluster:
```bash
chmod +x sh/test_all.sh
./sh/test_all.sh
```

3. Write results into a single CSV file:
```bash
chmod +x sh/create_results.sh
./sh/create_results.sh
```

---

## 🧪 Benchmarks

- Tested on the **University of Trento HPC cluster**
- Matrix size: 16,384 × 16,384 with 15% non-zero density
- Performance tested across configurations with **1 to 64 cores**
- Two placement strategies:
  - `pack:excl`: all processes on the same node
  - `scatter`: each process on a different node
- Detailed timing and memory usage graphs in the report

---

## 🧠 Insights

- **Reverse distribution** improves performance by avoiding element-wise index comparisons during multiplication.
- **Memory efficiency** is significantly better with CSR due to smaller index arrays.
- **Parallelizing inner loops** reduces thread initialization overhead and improves scalability.

---

## 📊 Results Overview

- **Reverse + CSR + Hybrid parallelization** gave the best performance overall.
- **CSR format** outperformed COO in both memory and execution time.
- **Hybrid parallelism** achieved the best speedup when properly balanced across nodes and cores.
- Reverse mode drastically reduced multiplication time by **eliminating index comparisons**.

---

## 📁 Folder Structure
```bash
optimized-matrix-multiplication/
├── csv_results/ # CSV files with benchmark or timing results
├── images/ # Plots, diagrams, and visual assets used in the report or README
├── related_papers/ # Reference papers and documentation
├── scripts/ # Python scripts to generate graphs
├── sh/ # Shell scripts for compilation and batch job submission
├── src/ # Source code implementations (C with MPI/OpenMP/hybrid)
```

