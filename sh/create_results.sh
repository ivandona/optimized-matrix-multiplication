#PBS -l select=1:ncpus=1:mem=1gb 
#PBS -l walltime=0:01:00
#PBS -q short_cpuQ
#PBS -o res/create_results.out
#PBS -e res/create_results.err

# Cartella dove si trovano i file csv e sh
res_dir="res"

# File finale nella directory corrente
csv_file="results.csv"

# Scrivi l'intestazione se il file non esiste
if [ ! -f "$csv_file" ]; then
    echo "type,n,density,processes,threads,total_time,generation_time,comm_total,comm_dist,comm_aggr,computation_total,computation_sort,computation_mult,sum_check" > "$csv_file"
fi

# Aggiungi contenuto di tutti i .csv presenti in res
for f in "$res_dir"/*.csv; do
    [ -e "$f" ] || continue  # Evita errori se non ci sono file
    cat "$f" >> "$csv_file"
done

# Elimina tutti i .csv e .sh nella cartella res
#rm -f "$res_dir"/*.csv "$res_dir"/*.sh

echo "✅ Unione completata in '$csv_file'. File temporanei rimossi."
