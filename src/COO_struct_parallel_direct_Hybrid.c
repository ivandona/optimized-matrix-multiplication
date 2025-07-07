//##############################################################################################################################
// Matrices Multiplication in format COO using a struct as representation (Hybrid direct version)
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
// Struct used for representation
//###############################################################

typedef struct {
    int row;
    int col;
    double value;
} COOElement;



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
void generate_sparse_matrix(COOElement** A, const int n, const int nnz){
    // allocation of matrix and support vectors
    *A = malloc(nnz * sizeof(COOElement));

    int* row_counts = calloc(n, sizeof(int));
    bool* used_cols = calloc(n, sizeof(bool));

    // check for allocation error
    if ((!(*A))||(!row_counts)||(!used_cols)){
        fprintf(stderr, "(Generation) Allocation error!\n");
        if(*A) free(*A);        //in theory the free(...) are not necessary due to the MPI_Abort
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

    int index = 0; //index of COO matrix, an array of struct
    for(i = 0; i < n; i++){
        memset(used_cols, 0, n * sizeof(bool));
        int j;
        for(j = 0; j < row_counts[i]; j++) {
            (*A)[index].row = i;
            (*A)[index].col = rand() % n;
            while(used_cols[(*A)[index].col]){
                (*A)[index].col = rand() % n;
            }
            used_cols[(*A)[index].col]=true;
            (*A)[index].value = rand() % 10 +1;
            index++;
        }
    }
    //free support vectors memory
    free(row_counts);
    free(used_cols);
}

//---------------------------------------------------------------
// (2.1) Compare function for qsort in function (2.2)
//---------------------------------------------------------------
int compare_row(const void* a, const void* b){
    return ((COOElement*)a)->row - ((COOElement*)b)->row;
}


//---------------------------------------------------------------
// (2.2) Compare function to sort the matrices by cols
//---------------------------------------------------------------
int* sort_by_column_threads(COOElement** A, const int n, const int nnz, const int num_threads){
    // to count how many elements per column
    int* count = calloc(n, sizeof(int));
    int* offset = malloc((n+1) * sizeof(int));
    if((!count)||(!offset)){
        printf("Sorting allocation error!\n");
        exit(1);
    }
    int i;
    #pragma omp parallel for num_threads(num_threads)
    for(i = 0; i < nnz; ++i){
        int col = (*A)[i].col;
        #pragma omp atomic
        count[col]++;
    }

    offset[0] = 0;
    offset[n] = nnz;
    for(i = 1; i < n; ++i){ //better to keep it serial (n is relatively small)
        offset[i] = offset[i - 1] + count[i - 1];
    }

    //allocate new array
    COOElement* A_final = (COOElement*)malloc(nnz * sizeof(COOElement));
    if(!A_final){
        printf("Sorting result allocation error!\n");
        exit(1);
    }

    //popolate the array
    #pragma omp parallel for num_threads(num_threads)
    for(i = 0; i < nnz; i++){
        int col = (*A)[i].col;
        int index;

        #pragma omp atomic capture
        index = --count[col];
        A_final[offset[col] + index] = (*A)[i];
    }
    //swap and free memory
    free(*A);
    (*A) = A_final;


    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<n; i++){
        int index = offset[i];
        int cnt = offset[i+1] - index;
        if(cnt>0){
            qsort(&((*A)[index]), cnt, sizeof(COOElement), compare_row);
        }
    }

    free(count);
    return offset;
}


//---------------------------------------------------------------
// (2.3) Compare function for qsort in function (2.4)
//---------------------------------------------------------------
int compare_col(const void* a, const void* b){
    return ((COOElement*)a)->col - ((COOElement*)b)->col;
}


//---------------------------------------------------------------
// (2.4) Compare function to sort the matrices by rows and cols
//---------------------------------------------------------------
int* sort_by_row_col_threads(COOElement* A, const int A_counts, const int rows, const int first_row, const int num_threads){
    int* A_offset = malloc((rows+1) * sizeof(int));
    if(!A_offset){
        printf("Sorting result allocation error!\n");
        exit(1);
    }

    // init first and last offsets  (very short, no need to parallelize)
    int i;
    for(i=0; i<=A[0].row-first_row; i++){
        A_offset[i]=0;
    }
    for(i=A[A_counts-1].row +1 -first_row; i<=rows; i++){
        A_offset[i]=A_counts;
    }
    
    // compute offsets
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<A_counts; i++){
        int prev = A[i-1].row;
        int row = A[i].row;
        if(prev < row){
            while(prev < row){
                A_offset[++prev -first_row]=i;
            }
        }
    }

    // sort all the sub array (A is already ordered by rows)
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<rows; i++){
        int index = A_offset[i];
        int count = A_offset[i+1] - index;
        if(count>0){
            qsort(&(A[index]), count, sizeof(COOElement), compare_col);
        }
    }
    return A_offset;
}


//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(COOElement** A, COOElement** B, int* local_A_count, const int nnz, const int n, const int world_rank, const int world_size, const int num_threads){
    //support variables for distribution
    int* A_counts = NULL;   //elements distribution
    int* A_displs = NULL;   //offset distribution
    COOElement* local_A = NULL; //local elements

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
            int x = (*A)[i].row;
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
        (*B) = malloc(nnz * sizeof(COOElement));
        if((*B)==NULL){
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }
    }
    local_A = malloc((*local_A_count) * sizeof(COOElement));
    if(local_A==NULL){
        fprintf(stderr, "(Distribution) Allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }


    //create custom MPI Datatype
    MPI_Datatype data_type;
    int block_lengths[3] = {1, 1, 1};
    MPI_Aint offsets[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

    MPI_Aint base_address;
    COOElement dummy;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&(dummy.row), &offsets[0]);
    MPI_Get_address(&(dummy.col), &offsets[1]);
    MPI_Get_address(&(dummy.value), &offsets[2]);

    offsets[0] -= base_address;
    offsets[1] -= base_address;
    offsets[2] -= base_address;

    MPI_Type_create_struct(3, block_lengths, offsets, types, &data_type);
    MPI_Type_commit(&data_type);


    //data distribution
    MPI_Scatterv((*A), A_counts, A_displs, data_type, local_A, (*local_A_count), data_type, 0, MPI_COMM_WORLD);
    MPI_Bcast((*B), nnz, data_type, 0, MPI_COMM_WORLD);

    //free & swap pointers (initial A and B not needed anymore)
    MPI_Type_free(&data_type);
    if(world_rank==0){
        free(A_counts);
        free(A_displs);
        free(*A);
    }
    (*A) = local_A;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO(COOElement* A, COOElement* B, int* A_offset, int* B_offset, const int rows, const int n, const int num_threads){    
    // allocation of final matrix
    double* C = calloc(n * rows, sizeof(double));
    if(!C){
        if(C) free(C);
        fprintf(stderr, "(Multiplication) allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // Multiplication: scroll A by rows and B by cols
    int A_row;
    #pragma omp parallel for num_threads(num_threads)
    for(A_row = 0; A_row<rows; A_row++){
        int B_col;
        for(B_col = 0; B_col<n; B_col++){
            int i = A_offset[A_row];
            int A_end = A_offset[A_row+1];
            int j = B_offset[B_col];
            int B_end = B_offset[B_col+1];
    
            while ((i<A_end)&&(j<B_end)){
                if (A[i].col<B[j].row){
                    i++;
                }
                else if(A[i].col>B[j].row){
                    j++;
                }
                else{
                    C[A_row*n +B_col] += A[i].value * B[j].value;
                    i++;
                    j++;
                }
            }
        }
    }
    free(A_offset);
    free(B_offset);
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
    COOElement* A = NULL;
    COOElement* B = NULL;


    //######################## 
    // matrices generation
    //########################
    if(world_rank == 0){
        start = MPI_Wtime();
        generate_sparse_matrix(&A, n, nnz);
        generate_sparse_matrix(&B, n, nnz);
        end = MPI_Wtime();
        cpu_time_generation = end - start;
    }



    //######################## 
    // matrices distribution
    //######################## 
    int A_counts;
    start = MPI_Wtime();
    comms_data_distribution(&A, &B, &A_counts, nnz, n, world_rank, world_size, num_threads);
    end = MPI_Wtime();
    comms_time_dist = end - start;



    //######################## 
    // sorting phase
    //######################## 
    // sorting B by cols
    // sorting A by rows and cols
    int* B_offset;
    int* A_offset;
    start = MPI_Wtime();
    int rows = n/world_size;
    int first_row = n-(world_size-world_rank)*rows;
    if(world_rank<n%world_size){
        rows++;
        first_row = world_rank*rows;
    }
    A_offset = sort_by_row_col_threads(A, A_counts, rows, first_row, num_threads);
    B_offset = sort_by_column_threads(&B, n, nnz, num_threads);
    end = MPI_Wtime();
    cpu_time_sorting = end - start;
    


    //######################## 
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_COO(A, B, A_offset, B_offset, rows, n, num_threads);
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
        printf("COO struct parallel direct Hybrid (n=%d, d=%.2f)[#pr=%d][#th=%d]\n", n, density_percentage/100, world_size, num_threads);
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
            snprintf(filename, sizeof(filename),"res/%s/output_COO_spd_Hybrid_%d_%d.csv", argv[2], world_size, num_threads);
        }
        else snprintf(filename, sizeof(filename),"res/output_COO_spd_Hybrid_%d_%d.csv", world_size, num_threads);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "COO_spd_Hybrid",         // algorithm
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
    free(A);
    free(B);
    if(world_rank==0){
        free(C);
    }
    
    MPI_Finalize();
    return 0;
}
