//##############################################################################################################################
// Matrices Multiplication in format COO using an array as representation (Hybrid reverse version)
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
//      (5) main
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
// (2) Function for sorting
//---------------------------------------------------------------
void sort_by_column_threads(int** row, int** col, double** val, const int n, const int nnz, const int num_threads){
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
    #pragma omp parallel for num_threads(num_threads)
    for(i=nnz-1; i >= 0; i--){
        int column = (*col)[i];
        int index;
        #pragma omp atomic capture
        index = --count[column];
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
    free(offset);
}


//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(int** A_row, int** A_col, double** A_val, int** B_row, int** B_col, double** B_val, int* local_A_count, int* local_B_count, const int nnz, const int n, const int world_rank, const int world_size, const int num_threads){
    //support variables for distribution
    int* A_counts = NULL;   //elements distribution
    int* B_counts = NULL;
    int* A_displs = NULL;   //offset distribution
    int* B_displs = NULL;

    int* local_A_row = NULL; //local elements
    int* local_A_col = NULL;
    double* local_A_val = NULL;
    int* local_B_row = NULL;
    int* local_B_col = NULL;
    double* local_B_val = NULL;

    //computing distribution among the processes
    if(world_rank==0){
        A_counts = calloc(world_size, sizeof(int));
        B_counts = calloc(world_size, sizeof(int));
        A_displs = calloc(world_size, sizeof(int));
        B_displs = calloc(world_size, sizeof(int));


        if((A_counts==NULL)||(B_counts==NULL)||(A_displs==NULL)||(B_displs==NULL)){
            if(A_counts) free(A_counts);
            if(B_counts) free(B_counts);
            if(A_displs) free(A_displs);
            if(B_displs) free(B_displs);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }

        //A_col and B_row distribution
        int i;
        int d = n / world_size;     //division (inteder)
        int r = n % world_size;     //rest (number of groups with one more element)
        int k = r * (d + 1);        //total number of elements in the largest groups
        #pragma omp parallel for num_threads(num_threads)
        for (i = 0; i < nnz; i++){
            //for A
            int x =(*A_col)[i];
            bool f = k>x;                           //larg or small group
            int g = (x - k*(!f) )/(d + f) + r*(!f); //group (or destination ptocess)
            #pragma omp atomic
            A_counts[g]++;

            //for B
            x = (*B_row)[i];
            f = k>x;                                //larg or small group
            g = (x - k*(!f) )/(d + f) + r*(!f);     //group (or destination ptocess)
            #pragma omp atomic
            B_counts[g]++;
        }


        for(i=1; i<world_size; i++){//better to keep it serial, world_size is relatively small (<=256)
            A_displs[i] = A_displs[i-1] + A_counts[i-1];
            B_displs[i] = B_displs[i-1] + B_counts[i-1];
        }
    }


    //counters distribution (to let each process know how much it'll recive)
    MPI_Scatter(A_counts, 1, MPI_INT, local_A_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_counts, 1, MPI_INT, local_B_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if((*local_A_count==0) || (*local_B_count==0)){
        fprintf(stderr, "local counter is zero, there is nothing to do!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //allocating the required space
    local_A_row = malloc((*local_A_count) * sizeof(int));
    local_A_col = malloc((*local_A_count) * sizeof(int));
    local_A_val = malloc((*local_A_count) * sizeof(double));
    local_B_row = malloc((*local_B_count) * sizeof(int));
    local_B_col = malloc((*local_B_count) * sizeof(int));
    local_B_val = malloc((*local_B_count) * sizeof(double));
    if((!local_A_row)||(!local_A_col)||(!local_A_val)||(!local_B_row)||(!local_B_col)||(!local_B_val)){
        if(local_A_row) free(local_A_row);
        if(local_A_col) free(local_A_col);
        if(local_A_val) free(local_A_val);
        if(local_B_row) free(local_B_row);
        if(local_B_col) free(local_B_col);
        if(local_B_val) free(local_B_val);
        fprintf(stderr, "(Distribution) Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //data distribution
    MPI_Scatterv((*A_row), A_counts, A_displs, MPI_INT, local_A_row, (*local_A_count), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_col), A_counts, A_displs, MPI_INT, local_A_col, (*local_A_count), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_val), A_counts, A_displs, MPI_DOUBLE, local_A_val, (*local_A_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*B_row), B_counts, B_displs, MPI_INT, local_B_row, (*local_B_count), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*B_col), B_counts, B_displs, MPI_INT, local_B_col, (*local_B_count), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*B_val), B_counts, B_displs, MPI_DOUBLE, local_B_val, (*local_B_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //free & swap pointers (initial A and B not needed anymore)
    if(world_rank==0){
        free(A_counts);
        free(B_counts);
        free(A_displs);
        free(B_displs);
        free(*A_row);
        free(*A_col);
        free(*A_val);
        free(*B_row);
        free(*B_col);
        free(*B_val);
    }
    (*A_row) = local_A_row;
    (*A_col) = local_A_col;
    (*A_val) = local_A_val;
    (*B_row) = local_B_row;
    (*B_col) = local_B_col;
    (*B_val) = local_B_val;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO_threads(int* A_row, int* A_col, double* A_val, int* B_row, int* B_col, double* B_val, const int A_counts, const int B_counts, const int n, const int num_threads){
    // allocation of final matrix
    double* C = calloc(n*n, sizeof(double));
    //allocation support vector
    int B_first_row = B_row[0];
    int B_last_row = B_row[B_counts-1];
    int* B_offset = malloc((B_last_row-B_first_row +2) * sizeof(int)); 
    if((!C)||(!B_offset)){
        fprintf(stderr, "(Multiplication) Allocation error!\n");
        if(C) free(C);
        if(B_offset) free(B_offset); 
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    B_offset[0] = 0;
    B_offset[B_last_row-B_first_row +1] = B_counts;


    //compute the offsets of B rows
    int i;
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<B_counts; i++){
        int prev = B_row[i-1] -B_first_row;
        int row = B_row[i] -B_first_row;
        if(prev < row){
            while(++prev <= row){
                B_offset[prev]=i;
            }
        }
    }


    //compute the multiplication
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<A_counts; i++){
        int col = A_col[i];
        if(B_first_row <= col && col <= B_last_row){ //check if col is in the B-rows range
            int j;
            for(j=B_offset[col-B_first_row]; j<B_offset[col+1-B_first_row]; j++){ //for all elements in that range
                int index = n*(A_row[i]) + B_col[j];
                double value = A_val[i] * B_val[j];
                #pragma omp atomic
                C[index] += value;
            }
        }
    }
    return C;
}


//---------------------------------------------------------------
// (5) Main function
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
    double cpu_time_sorting = 0;    //just root does the sorting
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
    // sorting phase
    //######################## 
    if(world_rank == 0){
        start = MPI_Wtime();
        // sorting A by cols
        sort_by_column_threads(&A_row, &A_col, &A_val, n, nnz, num_threads);
        end = MPI_Wtime();
        cpu_time_sorting = end - start;
    }


    //######################## 
    // matrices distribution
    //######################## 
    int A_counts;
    int B_counts;
    start = MPI_Wtime();
    comms_data_distribution(&A_row, &A_col, &A_val, &B_row, &B_col, &B_val, &A_counts, &B_counts, nnz, n, world_rank, world_size, num_threads);
    end = MPI_Wtime();
    comms_time_dist = end - start;


    //######################## 
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_COO_threads(A_row, A_col, A_val, B_row, B_col, B_val, A_counts, B_counts, n, num_threads);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //######################## 
    // matrices aggregation
    //########################
    double* C = NULL;
    start = MPI_Wtime();
    if(world_rank == 0){
        C = calloc(n * n, sizeof(double));
        if(!C){
            fprintf(stderr, "(Aggregation) allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }
    }
    MPI_Reduce(C_partial, C, n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    end = MPI_Wtime();
    free(C_partial);
    comms_time_aggr = end - start;
    comms_time_total = comms_time_dist + comms_time_aggr;

 
    //######################## 
    // final section
    //########################
    // print sections time for statistics
    if(world_rank == 0){
        printf("COO array parallel reverse Hybrid (n=%d, d=%.2f)[#pr=%d][#th=%d]\n", n, density_percentage/100, world_size, num_threads);
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
            snprintf(filename, sizeof(filename),"res/%s/output_COO_apr_Hybrid_%d_%d.csv", argv[2], world_size, num_threads);
        }
        else snprintf(filename, sizeof(filename),"res/output_COO_apr_Hybrid_%d_%d.csv", world_size, num_threads);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "COO_apr_Hybrid",         // algorithm
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
