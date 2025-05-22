//##############################################################################################################################
// Matrices Multiplication in format CSR using an array as representation (MPI reverse version)
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



//###############################################################
// Functions:
//---------------------------------------------------------------
//      (1) generation
//      (2) sorting (conversion)
//      (3) comms: data distribution
//      (4) multiplication
//      (5) main
//###############################################################

//---------------------------------------------------------------
// (1) Function for sparse matrix generation in COO format
//---------------------------------------------------------------
void generate_sparse_matrix(int** row_ptr, int** col, double** value, const int n, const int nnz){
    // allocation of matrix and support vectors
    *row_ptr = calloc(n+1, sizeof(int));
    *col = calloc(nnz, sizeof(int));
    *value = calloc(nnz, sizeof(double));

    int* row_counts = calloc(n, sizeof(int));
    bool* used_cols = calloc(n, sizeof(bool));

    // check for allocation error
    if((!*row_ptr) || (!*col) || (!*value) || (!row_counts) || (!used_cols)){
        printf("Allocation error!\n");
        if(*row_ptr) free(*row_ptr);
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

    int index = 0;
    (*row_ptr)[0] = 0;
    for(i = 0; i < n; i++){
        (*row_ptr)[i+1] = (*row_ptr)[i] + row_counts[i];
        memset(used_cols, 0, n * sizeof(bool));
        int j;
        for(j = 0; j < row_counts[i]; j++){
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
// (2) Function for sorting (conversion)
//---------------------------------------------------------------
void convert_CSR_to_CSC(int* row_ptr_csr, int* col_csr, double** val, int** col_ptr, int** row, const int n, const int nnz){
    // allocation of support and new vectors
    int* col_counts = calloc(n, sizeof(int));
    int* col_ptr_csc = malloc((n+1) * sizeof(int));
    int* row_csc = malloc(nnz * sizeof(int));
    double* val_csc = malloc(nnz * sizeof(double));

    if((!col_ptr_csc)||(!col_counts)||(!row_csc)||(!val_csc)){
        if(col_counts) free(col_counts);
        if(col_ptr_csc) free(col_ptr_csc);
        if(row_csc) free(row_csc);
        if(val_csc) free(val_csc);
        printf("Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    int i;
    for(i=0; i<nnz; i++){
        int col = col_csr[i];
        col_counts[col]++;
    }

    col_ptr_csc[0] = 0;
    for(i=0; i<n; i++){
        col_ptr_csc[i+1] = col_ptr_csc[i] + col_counts[i];
    }

    for(i=0; i<n; i++){
        int j;
        for(j=row_ptr_csr[i]; j<row_ptr_csr[i+1]; j++){
            int col = col_csr[j];
            int idx = col_ptr_csc[col] + (--col_counts[col]);
            row_csc[idx] = i;
            val_csc[idx] = (*val)[j];
        }
    }

    //swapping the two groups
    free(row_ptr_csr);
    free(col_csr);
    free(*val);

    (*val) = val_csc;
    (*col_ptr) = col_ptr_csc;
    (*row) = row_csc;

    // free memory
    free(col_counts);
}


//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(int** A_row, int** A_col_ptr, double** A_value, int** B_row_ptr, int** B_col, double** B_value, int local_rows, const int nnz, const int n, const int world_rank, const int world_size){
    //support variables for distribution
    int rows_per_process;   //basic distribution
    int local_A_count=0;    //counters
    int local_B_count=0;
    int* rows_distr=NULL;   //actual distribution
    int* rows_offset= NULL;
    int* A_counts = NULL;   //elements distribution
    int* B_counts = NULL;
    int* A_displs = NULL;   //offset distribution
    int* B_displs = NULL;

    int* local_A_row = NULL; //local elements
    int* local_A_col_ptr = NULL;
    double* local_A_val = NULL;
    int* local_B_row_ptr = NULL;
    int* local_B_col = NULL;
    double* local_B_val = NULL;

    //computing distribution among the processes
    if(world_rank==0){
        rows_per_process = n/world_size;
        rows_distr = malloc(world_size * sizeof(int));
        rows_offset = malloc(world_size * sizeof(int));
        A_counts = malloc(world_size * sizeof(int));
        B_counts = malloc(world_size * sizeof(int));
        A_displs = malloc(world_size * sizeof(int));
        B_displs = malloc(world_size * sizeof(int));


        if((rows_distr==NULL)||(rows_offset==NULL)||(A_counts==NULL)||(B_counts==NULL)||(A_displs==NULL)||(B_displs==NULL)){
            if(rows_distr) free(rows_distr); //in theory the free(...) are not necessary due to the MPI_Abort
            if(rows_offset) free(rows_offset);
            if(A_counts) free(A_counts);
            if(B_counts) free(B_counts);
            if(A_displs) free(A_displs);
            if(B_displs) free(B_displs);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }


        //A_col & B_row distribution and set displacement arrays
        int i;
        int rows_counter=0;
        for(i=0; i<world_size; i++){
            A_displs[i]=(*A_col_ptr)[rows_counter];
            B_displs[i]=(*B_row_ptr)[rows_counter];
            rows_offset[i] = rows_counter;

            rows_distr[i]=rows_per_process;
            if(i<n%world_size){
                rows_distr[i]++;
            }
            rows_counter += rows_distr[i];
        }

        //set counts arrays
        for(i=1; i<world_size; i++){
            A_counts[i-1]=A_displs[i]-A_displs[i-1];
            B_counts[i-1]=B_displs[i]-B_displs[i-1];
        }
        A_counts[i-1] = nnz-A_displs[i-1];
        B_counts[i-1] = nnz-B_displs[i-1];
    }

    //counters distribution (to let each process know how much it'll recive)
    MPI_Scatter(A_counts, 1, MPI_INT, &local_A_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_counts, 1, MPI_INT, &local_B_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if((local_A_count==0) || (local_B_count==0)){
        fprintf(stderr, "local counter is zero, there is nothing to do!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //allocating the required space
    local_A_row = malloc(local_A_count * sizeof(int));
    local_A_col_ptr = malloc((local_rows+1) * sizeof(int));
    local_A_val = malloc(local_A_count * sizeof(double));
    local_B_row_ptr = malloc((local_rows+1) * sizeof(int));
    local_B_col = malloc(local_B_count * sizeof(int));
    local_B_val = malloc(local_B_count * sizeof(double));

    if((!local_A_row)||(!local_A_col_ptr)||(!local_A_val)||(!local_B_row_ptr)||(!local_B_col)||(!local_B_val)){
        if(local_A_row) free(local_A_row);
        if(local_A_col_ptr) free(local_A_col_ptr);
        if(local_A_val) free(local_A_val);
        if(local_B_row_ptr) free(local_B_row_ptr);
        if(local_B_col) free(local_B_col);
        if(local_B_val) free(local_B_val);
        fprintf(stderr, "(Distribution) Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //data distribution (part 1)
    MPI_Scatterv((*A_row), A_counts, A_displs, MPI_INT, local_A_row, local_A_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_col_ptr), rows_distr, rows_offset, MPI_INT, local_A_col_ptr, local_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_value), A_counts, A_displs, MPI_DOUBLE, local_A_val, local_A_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*B_row_ptr), rows_distr, rows_offset, MPI_INT, local_B_row_ptr, local_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*B_col), B_counts, B_displs, MPI_INT, local_B_col, local_B_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*B_value), B_counts, B_displs, MPI_DOUBLE, local_B_val, local_B_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // convert global offsets to locals
    int i;
    for(i=local_rows-1; i>=0; i--){
        local_A_col_ptr[i]-=local_A_col_ptr[0];
        local_B_row_ptr[i]-=local_B_row_ptr[0];
    }
    local_A_col_ptr[local_rows] = local_A_count;
    local_B_row_ptr[local_rows] = local_B_count;


    //free & swap pointers (initial A and B not needed anymore)
    if(world_rank==0){
        free(rows_distr);
        free(rows_offset);
        free(A_counts);
        free(B_counts);
        free(A_displs);
        free(B_displs);
        free(*A_row);
        free(*A_col_ptr);
        free(*A_value);
        free(*B_row_ptr);
        free(*B_col);
        free(*B_value);
    }
    (*A_row) = local_A_row;
    (*A_col_ptr) = local_A_col_ptr;
    (*A_value) = local_A_val;
    (*B_row_ptr) = local_B_row_ptr;
    (*B_col) = local_B_col;
    (*B_value) = local_B_val;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_CSC_x_CRS(int* A_row, int* A_col_ptr, double* A_value, int* B_row_ptr, int* B_col, double* B_value, const int rows, const int n){
    // allocation of final matrix
    double* C = calloc(n*n, sizeof(double));
    if(!C){
        printf("Result allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // Multiplication: scroll A by cols and B by rows
    int k;
    for(k=0; k<rows; k++){
        int i,j;
        for(i=A_col_ptr[k]; i<A_col_ptr[k+1]; i++){
            for(j=B_row_ptr[k]; j<B_row_ptr[k+1]; j++){
                C[A_row[i]*n + B_col[j]] += A_value[i] * B_value[j];
            }
        }
    }
    return C;
}


//---------------------------------------------------------------
// (5) Main function
//---------------------------------------------------------------
int main(int argc, char** argv){
    //######################## 
    // initialization
    //########################
    MPI_Init(NULL, NULL);

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
    int* A_row_ptr = NULL;
    int* A_col = NULL;
    double* A_value = NULL;
    int* B_row_ptr = NULL;
    int* B_col = NULL;
    double* B_value = NULL;


    //########################
    // matrices generation
    //########################
    if(world_rank == 0){
        start = MPI_Wtime();
        generate_sparse_matrix(&A_row_ptr, &A_col, &A_value, n, nnz);
        generate_sparse_matrix(&B_row_ptr, &B_col, &B_value, n, nnz);
        end = MPI_Wtime();
        cpu_time_generation = end - start;
    }


    //########################
    // sorting phase
    //########################
    // converting A from CSR to CSC
    int* A_col_ptr = NULL;
    int* A_row = NULL;
    if(world_rank == 0){
        start = MPI_Wtime();
        convert_CSR_to_CSC(A_row_ptr, A_col, &A_value, &A_col_ptr, &A_row, n, nnz);
        end = MPI_Wtime();
        cpu_time_sorting = end - start;
    }
    

    //######################## 
    // matrices distribution
    //######################## 
    int local_rows = n/world_size;
    if(world_rank < n%world_size){
        local_rows++;
    }
    start = MPI_Wtime();
    comms_data_distribution(&A_row, &A_col_ptr, &A_value, &B_row_ptr, &B_col, &B_value, local_rows, nnz, n, world_rank, world_size);
    end = MPI_Wtime();
    comms_time_dist = end - start;


    //########################
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_CSC_x_CRS(A_row, A_col_ptr, A_value, B_row_ptr, B_col, B_value, local_rows, n);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime(); //used also for total_time
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
        printf("CSR array parallel reverse MPI (n=%d, d=%.2f)[#pr=%d]\n", n, density_percentage/100, world_size);
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
        int i,j;
        for(i=0; i<n; i++){
            for(j=0; j<n; j++){
                sum+=C[i*n +j];
            }
        }
        printf("Final C matrix, elements sum = %lf\n",sum);
        char filename[128];
        if(argc==2){
            snprintf(filename, sizeof(filename),"res/%s/output_CSR_apr_MPI_%d_%d.csv", argv[1], world_size, 1);
        }
        else snprintf(filename, sizeof(filename),"res/output_CSR_apr_MPI_%d_%d.csv", world_size, 1);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,1,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "CSR_apr_MPI",            // algorithm
                n,                        // matrix dimension
                density_percentage/100,   // matrix density
                world_size,               // process numbers
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
    free(A_col_ptr);
    free(A_value);
    free(B_row_ptr);
    free(B_col);
    free(B_value);
    if(world_rank==0){
        free(C);
    }

    MPI_Finalize();
    return 0;
}
