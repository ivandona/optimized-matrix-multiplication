# === Compiler flags ===
GCC = gcc
MPICC = mpicc
CFLAGS = -g -Wall -std=c99
OMP_FLAG = -fopenmp

# === Directories ===
SRC_DIR = src
BIN_DIR = build/bin
SRCS = $(wildcard $(SRC_DIR)/algorithms/base/*.c) \
       $(wildcard $(SRC_DIR)/algorithms/strassen/*.c)

# === Target names ===
TARGETS = $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%,$(SRCS))

# === Default target ===
all: $(TARGETS)

# === Rule to compile based on prefix ===
$(BIN_DIR)/%: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@if echo $(@F) | grep -q '^sequential_'; then \
		echo "[SEQ] Compiling $<..."; \
		$(GCC) $(CFLAGS) -o $@ $<; \
	elif echo $(@F) | grep -q '^mpi_'; then \
		echo "[MPI] Compiling $<..."; \
		$(MPICC) $(CFLAGS) -o $@ $<; \
	elif echo $(@F) | grep -q '^omp_'; then \
		echo "[OMP] Compiling $<..."; \
		$(GCC) $(CFLAGS) $(OMP_FLAG) -o $@ $<; \
	elif echo $(@F) | grep -q '^hybrid_'; then \
		echo "[HYBRID] Compiling $<..."; \
		$(MPICC) $(CFLAGS) $(OMP_FLAG) -o $@ $<; \
	else \
		echo "[??] Unknown prefix for $<"; \
		exit 1; \
	fi


# === Clean up ===
clean:
	rm -rf build
