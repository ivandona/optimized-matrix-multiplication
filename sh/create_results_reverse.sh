#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb 
#PBS -l walltime=0:01:00
#PBS -q short_cpuQ
#PBS -o res/create_results_reverse.out
#PBS -e res/create_results_reverse.err

# Directory principale contenente le sottocartelle numerate
res_dir="res"

# File finale nella directory corrente
csv_file="results_reverse.csv"

# Intestazione da usare
header="type,n,density,processes,threads,total_time,generation_time,comm_total,comm_dist,comm_aggr,computation_total,computation_sort,computation_mult,sum_check,nodes"

# Scrivi l’intestazione solo se il file non esiste
if [ ! -f "$csv_file" ]; then
    echo "$header" > "$csv_file"
fi

# Itera su tutte le sottocartelle numeriche in res/
for dir in "$res_dir"/*/; do
    # Estrai il nome della cartella (es. "4" da "res/4/")
    nodes=$(basename "$dir")
    # Verifica che sia un numero
    if [[ "$nodes" =~ ^[0-9]+$ ]]; then
        # Cerca i .csv nella sottocartella
        for f in "$dir"/*.csv; do
            [ -e "$f" ] || continue  # Evita errori se non ci sono file

            # Aggiungi ogni riga del csv al file finale, aggiungendo la colonna nodes
            awk -v nodes="$nodes" -F',' -v OFS=',' '{ print $0, nodes }' "$f" >> "$csv_file"
        done
    fi
done

echo "✅ Unione completata in '$csv_file'."
