//##############################################################################################################################
// Matrices Multiplication in format COO using a struct as representation (MPI direct version)
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
// (2) Compare functions to sort the matrices
//---------------------------------------------------------------
int compare_col(const void* a, const void* b){
    return ((COOElement*)a)->col - ((COOElement*)b)->col;
}

int compare_row_col(const void* a, const void* b){
    int a_row = ((COOElement*)a)->row;
    int b_row = ((COOElement*)b)->row;
    int result;
    if(a_row == b_row){
        result = ((COOElement*)a)->col - ((COOElement*)b)->col;
    }
    else{
        result = a_row - b_row;
    }
    return result;
}


//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(COOElement** A, COOElement** B, int* local_A_count, const int nnz, const int n, const int world_rank, const int world_size){
    //support variables for distribution
    int rows_per_process;    //basic distribution
    int* A_rows_distr=NULL;  //actual distribution
    int* A_counts = NULL;   //elements distribution
    int* A_displs = NULL;   //offset distribution

    COOElement* local_A = NULL; //local elements

    //computing distribution among the processes
    if(world_rank==0){
        rows_per_process = n/world_size;
        A_rows_distr = malloc(world_size * sizeof(int));
        A_counts = malloc(world_size * sizeof(int));
        A_displs = malloc(world_size * sizeof(int));


        if((A_rows_distr==NULL)||(A_counts==NULL)||(A_displs==NULL)){
            if(A_rows_distr) free(A_rows_distr); //in theory the free(...) are not necessary due to the MPI_Abort
            if(A_counts) free(A_counts);
            if(A_displs) free(A_displs);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }

        //A_row distribution
        int i;
        for(i=0; i<world_size; i++){
            A_rows_distr[i]=rows_per_process;
            if(i<n%world_size){
                A_rows_distr[i]++;
            }
        }

        //set counts and displacement arrays
        int jA=0;
        int A_rows_counter=A_rows_distr[0];
        A_displs[0]=0;

        for(i=0; i<nnz; i++){
            if((*A)[i].row>=A_rows_counter){
                jA++;
                A_displs[jA]=i;
                A_counts[jA-1]=A_displs[jA]-A_displs[jA-1];
                A_rows_counter+=A_rows_distr[jA];
            }
        }
        A_counts[jA]=nnz-A_displs[jA];
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
        free(A_rows_distr);
        free(A_counts);
        free(A_displs);
        free(*A);
    }
    (*A) = local_A;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO(COOElement* A, COOElement* B, const int A_counts, const int B_counts, const int n, const int world_rank, const int world_size){
    //support variables
    int rows = n/world_size;
    int first_row = n-(world_size-world_rank)*rows;
    if(world_rank<n%world_size){
        rows++;
        first_row = world_rank*rows;
    }
    
    // allocation of final matrix
    double* C = calloc(n * rows, sizeof(double));
    int* A_offset = calloc(rows + 1, sizeof(int));
    int* B_offset = calloc(n + 1, sizeof(int));
    if((!C)||(!A_offset)||(!B_offset)){
        if(C) free(C);
        if(A_offset) free(A_offset);
        if(B_offset) free(B_offset);
        fprintf(stderr, "(Multiplication) allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    //compute offsets for rows of A
    int i=1;
    int prev = A[0].row -first_row;
    while(i<A_counts){
        int row = A[i].row -first_row;
        while(prev < row){
            A_offset[++prev]=i;
        }
        i++;
    }
    while(prev<rows){
        A_offset[++prev]=A_counts;
    }

    //compute offsets for columns of B
    i=1;
    prev = B[0].col;
    while(i<B_counts){
        int col=B[i].col;
        if(prev<col){
            while (++prev<=col){
                B_offset[prev]=i;
            }
        }
        prev=col;
        i++;
    }
    while(++prev<=n){
        B_offset[prev]=B_counts;
    }

    // Multiplication: scroll A by rows and B by cols
    int j;
    int A_row;
    int B_col;

    for(A_row = 0; A_row<rows; A_row++){
        for(B_col = 0; B_col<n; B_col++){
            i = A_offset[A_row];
            int A_end = A_offset[A_row+1];
            j = B_offset[B_col];
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
    // sorting phase
    //######################## 
    // sorting B by cols
    if(world_rank == 0){
        start = MPI_Wtime();
        qsort(B, nnz, sizeof(COOElement), compare_col);
        end = MPI_Wtime();
        cpu_time_sorting = end - start;
    }


    //######################## 
    // matrices distribution
    //######################## 
    int A_counts;
    start = MPI_Wtime();
    comms_data_distribution(&A, &B, &A_counts, nnz, n, world_rank, world_size);
    end = MPI_Wtime();
    comms_time_dist = end - start;


    //######################## 
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    //sorting A by rows and cols
    qsort(A, A_counts, sizeof(COOElement), compare_row_col);
    end = MPI_Wtime();
    cpu_time_sorting += end - start;
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_COO(A, B, A_counts, nnz, n, world_rank, world_size);
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
        printf("COO struct parallel direct MPI (n=%d, d=%.2f)[#pr=%d]\n", n, density_percentage/100, world_size);
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
        if(argc==2){
            snprintf(filename, sizeof(filename),"res/%s/output_COO_spd_MPI_%d_%d.csv", argv[1], world_size, 1);
        }
        else snprintf(filename, sizeof(filename),"res/output_COO_spd_MPI_%d_%d.csv", world_size, 1);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,1,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "COO_spd_MPI",            // algorithm
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
    free(A);
    free(B);
    if(world_rank==0){
        free(C);
    }
    
    MPI_Finalize();
    return 0;
}
