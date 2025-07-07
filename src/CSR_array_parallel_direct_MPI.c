//##############################################################################################################################
// Matrices Multiplication in format CSR using an array as representation (MPI direct version)
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
//      (5) comms: data aggregation
//      (6) main
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
// (2.1) Function for sorting (conversion)
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

    for(i=n-1; i>=0; i--){
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
// (2.2) Function for sorting (CSR)
//---------------------------------------------------------------
void sorting_CSR(int* row_ptr, int** col, double** val, const int n, const int nnz){
    int i;
    for(i=0; i<n; i++){
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        int* idx = calloc(n, sizeof(int));
        if(!idx){
            printf("Sorting allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }

        int j;
        for(j=start; j<end; j++){
            idx[(*col)[j]]=1;
        }

        int count=0;
        for(j=0; j<n; j++){
            if(idx[j]){
                idx[j] = count++;
            }
        }

        int col_tmp1, col_tmp2;
        double val_tmp1, val_tmp2;
        int index;
        for(j=start; j<end; j++){
            col_tmp1 = (*col)[j];
            val_tmp1 = (*val)[j];
            index = idx[col_tmp1];

            while(index != -1){
                idx[col_tmp1] = -1;
                col_tmp2 = (*col)[start + index];
                val_tmp2 = (*val)[start + index];
                (*col)[start + index] = col_tmp1;
                (*val)[start + index] = val_tmp1;
                index = idx[col_tmp2];

                if(index != -1){
                    idx[col_tmp1] = -1;
                    col_tmp1 = (*col)[start + index];
                    val_tmp1 = (*val)[start + index];
                    (*col)[start + index] = col_tmp2;
                    (*val)[start + index] = val_tmp2;
                    index = idx[col_tmp1];
                }
            }
        }
        free(idx);
    }
}

//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(int** A_row_ptr, int** A_col, double** A_value, int** B_row, int** B_col_ptr, double** B_value, int local_rows, const int nnz, const int n, const int world_rank, const int world_size){
    //support variables for distribution
    int rows_per_process;   //basic distribution
    int local_A_count=0;    //counters
    int* rows_distr=NULL;   //actual distribution
    int* rows_offset= NULL;
    int* A_counts = NULL;   //elements distribution
    int* A_displs = NULL;   //offset distribution

    int* local_A_row_ptr = NULL; //local elements
    int* local_A_col = NULL;
    double* local_A_val = NULL;

    //computing distribution among the processes
    if(world_rank==0){
        rows_per_process = n/world_size;
        rows_distr = malloc(world_size * sizeof(int));
        rows_offset = malloc(world_size * sizeof(int));
        A_counts = malloc(world_size * sizeof(int));
        A_displs = malloc(world_size * sizeof(int));


        if((rows_distr==NULL)||(rows_offset==NULL)||(A_counts==NULL)||(A_displs==NULL)){
            if(rows_distr) free(rows_distr); //in theory the free(...) are not necessary due to the MPI_Abort
            if(rows_offset) free(rows_offset);
            if(A_counts) free(A_counts);
            if(A_displs) free(A_displs);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }

        //A_rows distribution
        //set displacement arrays
        int i;
        int rows_counter=0;
        for(i=0; i<world_size; i++){
            A_displs[i]=(*A_row_ptr)[rows_counter];
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
        }
        A_counts[i-1] = nnz-A_displs[i-1];
    }

    //counters distribution (to let each process know how much it'll recive)
    MPI_Scatter(A_counts, 1, MPI_INT, &local_A_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(local_A_count==0){
        fprintf(stderr, "local counter is zero, there is nothing to do!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //allocating the required space
    if(world_rank!=0){
        (*B_row) = malloc(nnz * sizeof(int));
        (*B_col_ptr) = malloc((n+1) * sizeof(int));
        (*B_value) = malloc(nnz * sizeof(double));
        if((*B_row)==NULL || (*B_col_ptr)==NULL || (*B_value)==NULL){
            if(*B_row) free(*B_row);
            if(*B_col_ptr) free(*B_col_ptr);
            if(*B_value) free(*B_value);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }
    }

    local_A_row_ptr = malloc((local_rows+1) * sizeof(int));
    local_A_col = malloc(local_A_count * sizeof(int));
    local_A_val = malloc(local_A_count * sizeof(double));
    if((!local_A_row_ptr)||(!local_A_col)||(!local_A_val)){
        if(local_A_row_ptr) free(local_A_row_ptr);
        if(local_A_col) free(local_A_col);
        if(local_A_val) free(local_A_val);
        fprintf(stderr, "(Distribution) Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //data distribution (part 1)
    MPI_Scatterv((*A_row_ptr), rows_distr, rows_offset, MPI_INT, local_A_row_ptr, local_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_col), A_counts, A_displs, MPI_INT, local_A_col, local_A_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((*A_value), A_counts, A_displs, MPI_DOUBLE, local_A_val, local_A_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B_row), nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B_col_ptr), n+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B_value), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // convert global offsets to locals
    int i;
    for(i=local_rows-1; i>=0; i--){
        local_A_row_ptr[i]-=local_A_row_ptr[0];
    }
    local_A_row_ptr[local_rows] = local_A_count;


    //free & swap pointers (initial A and B not needed anymore)
    if(world_rank==0){
        free(rows_distr);
        free(rows_offset);
        free(A_counts);
        free(A_displs);
        free(*A_row_ptr);
        free(*A_col);
        free(*A_value);
    }
    (*A_row_ptr) = local_A_row_ptr;
    (*A_col) = local_A_col;
    (*A_value) = local_A_val;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_CSC_x_CRS(int* A_row_ptr, int* A_col, double* A_value, int* B_row, int* B_col_ptr, double* B_value, const int local_rows, const int n){
    // allocation of final matrix
    double* C = calloc(n*local_rows, sizeof(double));
    if(!C){
        printf("Result allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // Multiplication: scroll A by rows and B by cols
    int A_row;
    for(A_row = 0; A_row<local_rows; A_row++){
        int B_col;
        for(B_col = 0; B_col<n; B_col++){
            int i = A_row_ptr[A_row];
            int A_row_end = A_row_ptr[A_row+1];
            int j = B_col_ptr[B_col];
            int B_col_end = B_col_ptr[B_col+1];
    
            while ((i<A_row_end)&&(j<B_col_end)){
                if(A_col[i]<B_row[j]){
                    i++;
                }
                else if(A_col[i]>B_row[j]){
                    j++;
                }
                else{
                    C[A_row*n +B_col] += A_value[i] * B_value[j];
                    i++;
                    j++;
                }
            }
        }
    }
    return C;
}


//---------------------------------------------------------------
// (5) Data Aggregation function
//---------------------------------------------------------------
double* comms_data_aggregation(double* C_partial, const int local_rows, const int n, const int world_rank, const int world_size){
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
        for(i=0; i<world_size; i++){
            C_counts[i]=rows_per_process*n;
            if(i<n%world_size){
                C_counts[i]+=n;
            }
            if(i>0){
                C_displs[i] = C_displs[i-1] + C_counts[i-1];
            }
        }
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
    int* B_col_ptr = NULL;
    int* B_row = NULL;
    if(world_rank == 0){
        start = MPI_Wtime();
        convert_CSR_to_CSC(B_row_ptr, B_col, &B_value, &B_col_ptr, &B_row, n, nnz);
        // sorting A by row and col
        sorting_CSR(A_row_ptr, &A_col, &A_value, n, nnz);
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
    comms_data_distribution(&A_row_ptr, &A_col, &A_value, &B_row, &B_col_ptr, &B_value, local_rows, nnz, n, world_rank, world_size);
    end = MPI_Wtime();
    comms_time_dist = end - start;
    
    
    //########################
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_CSC_x_CRS(A_row_ptr, A_col, A_value, B_row, B_col_ptr, B_value, local_rows, n);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime(); //used also for total_time
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;
   

    //######################## 
    // matrices aggregation
    //########################
    double* C = NULL;
    start = MPI_Wtime();
    C = comms_data_aggregation(C_partial, local_rows, n, world_rank, world_size);
    end = MPI_Wtime();
    free(C_partial);
    comms_time_aggr = end - start;
    comms_time_total = comms_time_dist + comms_time_aggr;


    //########################
    // final section
    //########################
    // print sections time for statistics
    if(world_rank == 0){
        printf("CSR array parallel direct MPI (n=%d, d=%.2f)[#pr=%d]\n", n, density_percentage/100, world_size);
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
            snprintf(filename, sizeof(filename),"res/%s/output_CSR_apd_MPI_%d_%d.csv", argv[1], world_size, 1);
        }
        else snprintf(filename, sizeof(filename),"res/output_CSR_apd_MPI_%d_%d.csv", world_size, 1);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,1,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "CSR_apd_MPI",            // algorithm
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
    free(A_row_ptr);
    free(A_col);
    free(A_value);
    free(B_row);
    free(B_col_ptr);
    free(B_value);
    if(world_rank==0){
        free(C);
    }
    
    MPI_Finalize();
    return 0;
}
