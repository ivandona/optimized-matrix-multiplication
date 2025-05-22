//##############################################################################################################################
// Matrices Multiplication in format COO using an array as representation (Hybrid direct version)
//##############################################################################################################################
//###############################################################
// Libraries
//###############################################################
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


//###############################################################
// Functions:
//---------------------------------------------------------------
//      (1) generation
//      (2) comparison (for sorting)
//      (3) comms: data distribution
//      (4) multiplication
//      (5) comms: data aggregation
//      (6) main
//###############################################################

//---------------------------------------------------------------
// (1) Function for sparse matrix generation in COO format
//---------------------------------------------------------------
void generate_sparse_matrix(int** row, int** col, double** value, const int n, const int nnz){
    // allocation of matrix and support vectors
    *row = calloc(nnz, sizeof(int));
    *col = calloc(nnz, sizeof(int));
    *value = calloc(nnz, sizeof(double));

    int* row_counts = calloc(n, sizeof(int));
    bool* used_cols = calloc(n, sizeof(bool));

    // check for allocation error
    if((!*row)||(!*col)||(!*value)||(!row_counts)||(!used_cols)){
        printf("(Generation) Allocation error!\n");
        if(*row) free(*row);
        if(*col) free(*col);
        if(*value) free(*value);
        if(row_counts) free(row_counts);
        if(used_cols) free(used_cols);
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // Random elements distribution along the rows
    int i;
    for(i = 0; i < nnz; i++){
        int k = rand() % n;
        while(row_counts[k]==n){
            k = rand() % n;
        }
        row_counts[k]++;
    }

    int index = 0; //index of COO matrix, three arrays
    for(i = 0; i < n; i++){
        memset(used_cols, 0, n * sizeof(bool));
        int j;
        for(j = 0; j < row_counts[i]; j++){
            (*row)[index] = i;
            (*col)[index] = rand() % n;
            while(used_cols[(*col)[index]]){
                (*col)[index] = rand() % n;
            }
            used_cols[(*col)[index]]=true;
            (*value)[index] = rand() % 10 +1;
            index++;
        }
    }
    //free support vectors memory
    free(row_counts);
    free(used_cols);
}

//---------------------------------------------------------------
// (2.1) Compare function to sort the matrices by cols
//---------------------------------------------------------------
int* sort_by_column_threads(int** row, int** col, double** val, const int n, const int nnz, const int num_threads){
    // allocation of support vectors
    int* count = calloc(n, sizeof(int));
    int* offset = malloc((n+1) * sizeof(int));
    int* idx = malloc(nnz * sizeof(int));
    if((!count)||(!offset)||(!idx)){
        if(count) free(count);
        if(offset) free(offset);
        if(idx) free(idx);
        printf("Sorting allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize(); 
        exit(1);
    }

    // Initialize idx with {0, 1, 2, ..., n-1}
    int i;
    #pragma omp parallel for num_threads(num_threads)
    for(i = 0; i < nnz; ++i){
        int column = (*col)[i];
        #pragma omp atomic
        count[column]++;
    }

    offset[0] = 0;
    offset[n] = nnz;
    for(i = 1; i < n; ++i){ //better to keep it serial (n is relatively small)
        offset[i] = offset[i - 1] + count[i - 1];
    }
    
    // Sort idx[] based on A_col[]
    for(i=nnz-1; i >= 0; i--){
        int column = (*col)[i];
        int index = --count[column];
        idx[offset[column] + index] = i;
    }

    // allocation of new arrays
    int* row_sorted = malloc(nnz * sizeof(int));
    int* col_sorted = malloc(nnz * sizeof(int));
    double* val_sorted = malloc(nnz * sizeof(double));

    if (!row_sorted || !col_sorted || !val_sorted){
        if(row_sorted) free(row_sorted);
        if(col_sorted) free(col_sorted);
        if(val_sorted) free(val_sorted);
        printf("Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //sorting the arrays based on idx
    #pragma omp parallel for num_threads(num_threads)
    for(i = 0; i < nnz; i++){
        row_sorted[i] = (*row)[idx[i]];
        col_sorted[i] = (*col)[idx[i]];
        val_sorted[i] = (*val)[idx[i]];
    }
    //swapping the two groups
    free(*row);
    free(*col);
    free(*val);

    (*row) = row_sorted;
    (*col) = col_sorted;
    (*val) = val_sorted;

    // free memory
    free(idx);
    return offset;
}


//---------------------------------------------------------------
// (2.2) Compare function to sort the matrices by rows and cols
//---------------------------------------------------------------
int* sort_by_row_col_threads(int** row, int** col, double** val, const int first_row, const int rows, const int count, const int n, const int num_threads){
    int* offset = malloc((rows+1) * sizeof(int));
    if(!offset){
        printf("Sorting result allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // init first and last offsets  (very short, no need to parallelize)
    int i;
    for(i=0; i<=(*row)[0]-first_row; i++){
        offset[i]=0;
    }
    for(i=(*row)[count-1] +1 -first_row; i<=rows; i++){
        offset[i]=count;
    }

    // compute offsets
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<count; i++){
        int prev = (*row)[i-1];
        int r = (*row)[i];
        if(prev < r){
            while(prev < r){
                offset[++prev -first_row]=i;
            }
        }
    }

    // sort all the sub array (A is already ordered by rows)
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<rows; i++){
        int index = offset[i];
        int count = offset[i+1] - index;
        if(count>0){
            int* idx = calloc(n, sizeof(int));
            int* col_tmp = malloc(count * sizeof(int));
            double* val_tmp = malloc(count * sizeof(double));
            if((!idx)||(!col_tmp)||(!val_tmp)){
                if(idx) free(idx);
                if(col_tmp) free(col_tmp);
                if(val_tmp) free(val_tmp);
                fprintf(stderr, "(Sorting) Allocation error!\n");
                exit(1);
            }
            int k;
            for(k=0; k<count; k++){
                idx[(*col)[index + k]]++;
            }
            int counter=0;
            for(k=0; k<n; k++){
                if(idx[k]){
                    idx[k] = counter;
                    counter++;
                }
            }
            int index2;
            for(k=0; k<count; k++){
                index2 = idx[(*col)[index + k]];
                col_tmp[index2] = (*col)[index + k];
                val_tmp[index2] = (*val)[index + k];
            }
            memcpy((*col) + index, col_tmp, count * sizeof(int));
            memcpy((*val) + index, val_tmp, count * sizeof(double));

            free(idx);
            free(col_tmp);
            free(val_tmp);
        }
    }
    return offset;
}


//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(int** A_row, int** A_col, double** A_val, int** B_row, int** B_col, double** B_val, int* local_A_count, const int nnz, const int n, const int world_rank, const int world_size, const int num_threads){
    //support variables for distribution
    int* A_counts = NULL;   //elements distribution
    int* A_displs = NULL;   //offset distribution

    int* local_A_row = NULL; //local elements
    int* local_A_col = NULL;
    double* local_A_val = NULL;

    //computing distribution among the processes
    if(world_rank==0){
        A_counts = calloc(world_size, sizeof(int));
        A_displs = calloc(world_size, sizeof(int));


        if((A_counts==NULL)||(A_displs==NULL)){
            if(A_counts) free(A_counts);
            if(A_displs) free(A_displs);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }

        //A_row distribution
        int i;
        int d = n / world_size;     //division (inteder)
        int r = n % world_size;     //rest (number of groups with one more element)
        int k = r * (d + 1);        //total number of elements in the largest groups
        #pragma omp parallel for num_threads(num_threads)
        for (i = 0; i < nnz; i++){
            //for A
            int x = (*A_row)[i];
            bool f = k>x;                                //larg or small group
            int g = (x - k*(!f) )/(d + f) + r*(!f);     //group (or destination ptocess)
            #pragma omp atomic
            A_counts[g]++;
        }


        for(i=1; i<world_size; i++){//better to keep it serial, world_size is relatively small (<=256)
            A_displs[i] = A_displs[i-1] + A_counts[i-1];
        }
    }


    //counter distribution (to let each process know how much it'll recive)
    MPI_Scatter(A_counts, 1, MPI_INT, local_A_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(*local_A_count==0){
        fprintf(stderr, "local counter is zero, there is nothing to do!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //allocating the required space
    if(world_rank!=0){
        (*B_row) = malloc(nnz * sizeof(int));
        (*B_col) = malloc(nnz * sizeof(int));
        (*B_val) = malloc(nnz * sizeof(double));
        if((*B_row)==NULL || (*B_col)==NULL || (*B_val)==NULL){
            if(*B_row) free(*B_row);
            if(*B_col) free(*B_col);
            if(*B_val) free(*B_val);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }
    }
    local_A_row = malloc((*local_A_count) * sizeof(int));
    local_A_col = malloc((*local_A_count) * sizeof(int));
    local_A_val = malloc((*local_A_count) * sizeof(double));
    if((!local_A_row)||(!local_A_col)||(!local_A_val)){
        if(local_A_row) free(local_A_row);
        if(local_A_col) free(local_A_col);
        if(local_A_val) free(local_A_val);
        fprintf(stderr, "(Distribution) Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //data distribution
    MPI_Scatterv((*A_row), A_counts, A_displs, MPI_INT, local_A_row, (*local_A_count), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_col), A_counts, A_displs, MPI_INT, local_A_col, (*local_A_count), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_val), A_counts, A_displs, MPI_DOUBLE, local_A_val, (*local_A_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B_row), nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B_col), nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B_val), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //free & swap pointers (initial A and B not needed anymore)
    if(world_rank==0){
        free(A_counts);
        free(A_displs);
        free(*A_row);
        free(*A_col);
        free(*A_val);
    }
    (*A_row) = local_A_row;
    (*A_col) = local_A_col;
    (*A_val) = local_A_val;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO_threads(int* A_row_offset, int* A_col, double* A_val, int* B_row, int* B_col_offset, double* B_val, const int A_rows, const int n, const int num_threads){
    // allocation of final matrix
    double* C = calloc(n*A_rows, sizeof(double));
    if(!C){
        fprintf(stderr, "(Multiplication) allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // Multiplication: scroll A by rows and B by cols
    int A_row;
    #pragma omp parallel for num_threads(num_threads)
    for(A_row = 0; A_row<A_rows; A_row++){
        int B_col;
        for(B_col = 0; B_col<n; B_col++){
            int i = A_row_offset[A_row];
            int A_end = A_row_offset[A_row+1];
            int j = B_col_offset[B_col];
            int B_end = B_col_offset[B_col+1];
    
            while ((i<A_end)&&(j<B_end)){
                if(A_col[i]<B_row[j]){
                    i++;
                }
                else if(A_col[i]>B_row[j]){
                    j++;
                }
                else{
                    C[A_row*n +B_col] += A_val[i] * B_val[j];
                    i++;
                    j++;
                }
            }
        }
    }
    free(A_row_offset);
    free(B_col_offset);
    return C;
}

//---------------------------------------------------------------
// (5) Data Aggregation function
//---------------------------------------------------------------
double* comms_data_aggregation(double* C_partial, const int n, const int world_rank, const int world_size){
    //array ans support variables to receive the data
    double* C = NULL;
    int* C_displs = NULL;
    int* C_counts = NULL;
    if(world_rank == 0){
        C = calloc(n * n, sizeof(double));
        C_displs = malloc(world_size * sizeof(int));
        C_counts = malloc(world_size * sizeof(int));

        if((!C)||(!C_displs)||(!C_counts)){
            if(C) free(C);
            if(C_displs) free(C_displs);
            if(C_counts) free(C_counts);
            fprintf(stderr, "(Aggregation) allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }

        int rows_per_process = n/world_size;
        int i;
        C_displs[0]=0;
        for(i=0; i<world_size; i++){ //relatively small
            C_counts[i]=rows_per_process*n;
            if(i<n%world_size){
                C_counts[i]+=n;
            }
            if(i>0){
                C_displs[i] = C_displs[i-1] + C_counts[i-1];
            }
        }
    }
    int local_rows = n/world_size;
    if(world_rank<n%world_size){
        local_rows++;
    }
    MPI_Gatherv(C_partial, local_rows*n, MPI_DOUBLE, C, C_counts, C_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(world_rank == 0){
        free(C_displs);
        free(C_counts);
    }
    return C;
}


//---------------------------------------------------------------
// (6) Main function
//---------------------------------------------------------------
int main(int argc, char** argv){
    if(argc<2){
        printf("Error, number of threads not specified");
        exit(1);
    }

    //######################## 
    // initialization
    //########################
    MPI_Init(NULL, NULL);
    const int num_threads = atoi(argv[1]);

    // total time
    double start_tot = MPI_Wtime();

    // get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        fprintf(stderr, "Error, at least 2 processes required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }
	
	// get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    //######################## 
    // main variables
    //########################
    int n = 16384;  // Dimension of the matrices (n*n)
    double density_percentage = 15;  // Percentage of non null elements (es. 15 -> 15%)
    int nnz = (n * n * density_percentage/100);  // Actual non null elements

    double start, end;
    double cpu_time_generation;
    double cpu_time_computation;    //divided in sorting and multiplying
    double cpu_time_sorting;    
    double cpu_time_multiplying;
    double comms_time_total;        //divided in distribution and aggregation
    double comms_time_dist;
    double comms_time_aggr;

    srand(40);//(time(NULL)); //a seed is used to limit randomization to improve comparison accuracy

    // matrices variables
    int* A_row = NULL;
    int* A_col = NULL;
    double* A_val = NULL;
    int* B_row = NULL;
    int* B_col = NULL;
    double* B_val = NULL;


    //######################## 
    // matrices generation
    //########################
    if(world_rank == 0){
        start = MPI_Wtime();
        generate_sparse_matrix(&A_row, &A_col, &A_val, n, nnz);
        generate_sparse_matrix(&B_row, &B_col, &B_val, n, nnz);
        end = MPI_Wtime();
        cpu_time_generation = end - start;
    }



    //######################## 
    // matrices distribution
    //######################## 
    int A_counts;
    start = MPI_Wtime();
    comms_data_distribution(&A_row, &A_col, &A_val, &B_row, &B_col, &B_val, &A_counts, nnz, n, world_rank, world_size, num_threads);
    end = MPI_Wtime();
    comms_time_dist = end - start;


    //######################## 
    // sorting phase
    //######################## 
    // sorting B by cols and A by rows and cols
    start = MPI_Wtime();
    int A_rows = n/world_size;
    int A_first_row = n-(world_size-world_rank)*A_rows;
    if(world_rank<n%world_size){
        A_rows++;
        A_first_row = world_rank*A_rows;
    }
    int* A_rows_offset = sort_by_row_col_threads(&A_row, &A_col, &A_val, A_first_row, A_rows, A_counts, n, num_threads);
    int* B_col_offset = sort_by_column_threads(&B_row, &B_col, &B_val, n, nnz, num_threads);
    end = MPI_Wtime();
    cpu_time_sorting = end - start;




    //######################## 
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_COO_threads(A_rows_offset, A_col, A_val, B_row, B_col_offset, B_val, A_rows, n, num_threads);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //######################## 
    // matrices aggregation
    //########################
    double* C = NULL;
    start = MPI_Wtime();
    C = comms_data_aggregation(C_partial, n, world_rank, world_size);
    end = MPI_Wtime();
    free(C_partial);
    comms_time_aggr = end - start;
    comms_time_total = comms_time_dist + comms_time_aggr;

 
    //######################## 
    // final section
    //########################
    // print sections time for statistics
    if(world_rank == 0){
        printf("COO array parallel direct Hybrid (n=%d, d=%.2f)[#pr=%d][#th=%d]\n", n, density_percentage/100, world_size, num_threads);
        printf("Total time: %f seconds\n", end - start_tot);
        printf("Generation time: %f seconds\n", cpu_time_generation);
        printf("Communication time (total): %f seconds\n", comms_time_total);
        printf("Communication time (dist): %f seconds\n", comms_time_dist);
        printf("Communication time (aggr): %f seconds\n", comms_time_aggr);
        printf("Computation time (total): %f seconds\n", cpu_time_computation);
        printf("Computation time (sorting): %f seconds\n", cpu_time_sorting);
        printf("Computation time (multiplication): %f seconds\n", cpu_time_multiplying);

        // computing the sum of all elements to check code correctness (require fix seed for rand())
        double sum=0;
        int i, j;
        for(i=0; i<n; i++){
            for(j=0; j<n; j++){
                sum+=C[i*n +j];
            }
        }
        printf("Final C matrix, elements sum = %lf\n",sum);
        char filename[128];
        if(argc==3){
            snprintf(filename, sizeof(filename),"res/%s/output_COO_apd_Hybrid_%d_%d.csv", argv[2], world_size, num_threads);
        }
        else snprintf(filename, sizeof(filename),"res/output_COO_apd_Hybrid_%d_%d.csv", world_size, num_threads);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "COO_apd_Hybrid",         // algorithm
                n,                        // matrix dimension
                density_percentage/100,   // matrix density
                world_size,               // process numbers
                num_threads,              // thread numbers
                end - start_tot,          // total time
                cpu_time_generation,      // generation time
                comms_time_total,         // total communication time
                comms_time_dist,          // comm. distribuzione time
                comms_time_aggr,          // comm. aggregation time
                cpu_time_computation,     // total computation time
                cpu_time_sorting,         // comp. sorting time
                cpu_time_multiplying,     // comp. multiplication time
                sum                       // sum (to check correctness)
            );
            fclose(fp);
        }
    }

    //free all memory
    free(A_row);
    free(A_col);
    free(A_val);
    free(B_row);
    free(B_col);
    free(B_val);
    if(world_rank==0){
        free(C);
    }

    MPI_Finalize();
    return 0;
}
