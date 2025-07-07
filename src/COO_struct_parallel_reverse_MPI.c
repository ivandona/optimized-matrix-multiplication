//##############################################################################################################################
// Matrices Multiplication in format COO using a struct as representation (MPI reverse version)
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
//      (5) main
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
// (2) Compare function to sort the matrices
//---------------------------------------------------------------
int compare_col(const void* a, const void* b) {
    return ((COOElement*)a)->col - ((COOElement*)b)->col;
}


//---------------------------------------------------------------
// (3) Data distribution function
//---------------------------------------------------------------
void comms_data_distribution(COOElement** A, COOElement** B, int* local_A_count, int* local_B_count, const int nnz, const int n, const int world_rank, const int world_size){
    //support variables for distribution
    int row_per_process;    //basic distribution
    int* A_col_distr=NULL;  //actual distribution
    int* A_counts = NULL;   //elements distribution
    int* B_counts = NULL;
    int* A_displs = NULL;   //offset distribution
    int* B_displs = NULL;
    COOElement* local_A = NULL; //local elements
    COOElement* local_B = NULL;

    //computing distribution among the processes
    if(world_rank==0){
        row_per_process = n/world_size;
        A_col_distr = malloc(world_size * sizeof(int));
        A_counts = calloc(world_size, sizeof(int));
        B_counts = calloc(world_size, sizeof(int));
        A_displs = calloc(world_size, sizeof(int));
        B_displs = calloc(world_size, sizeof(int));


        if((A_col_distr==NULL)||(A_counts==NULL)||(B_counts==NULL)||(A_displs==NULL)||(B_displs==NULL)){
            if(A_col_distr) free(A_col_distr); //in theory the free(...) are not necessary due to the MPI_Abort
            if(A_counts) free(A_counts);
            if(B_counts) free(B_counts);
            if(A_displs) free(A_displs);
            if(B_displs) free(B_displs);
            fprintf(stderr, "(Distribution) Allocation error!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            exit(1);
        }


        //A_col & B_row distribution
        int i;
        for(i=0; i<world_size; i++){
            A_col_distr[i]=row_per_process;
            if(i<n%world_size){
                A_col_distr[i]++;
            }
        }

        //set counts and displacement arrays
        int jA=0;
        int A_col_counter=A_col_distr[0];
        int jB=0;
        int B_row_counter=A_col_distr[0];

        for(i=0; i<nnz; i++){
            if((*A)[i].col>=A_col_counter){
                jA++;
                A_displs[jA]=i;
                A_counts[jA-1]=A_displs[jA]-A_displs[jA-1];
                A_col_counter+=A_col_distr[jA];
            }
            if((*B)[i].row>=B_row_counter){
                jB++;
                B_displs[jB]=i;
                B_counts[jB-1]=B_displs[jB]-B_displs[jB-1];
                B_row_counter+=A_col_distr[jB];
            }
        }
        A_counts[jA]=nnz-A_displs[jA];
        B_counts[jB]=nnz-B_displs[jB];
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
    local_A = malloc((*local_A_count) * sizeof(COOElement));
    local_B = malloc((*local_B_count) * sizeof(COOElement));
    if((local_A==NULL)||(local_B==NULL)){
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
    MPI_Scatterv((*B), B_counts, B_displs, data_type, local_B, (*local_B_count), data_type, 0, MPI_COMM_WORLD);

    //free & swap pointers (initial A and B not needed anymore)
    MPI_Type_free(&data_type);
    if(world_rank==0){
        free(A_col_distr);
        free(A_counts);
        free(B_counts);
        free(A_displs);
        free(B_displs);
        free(*A);
        free(*B);
    }
    (*A) = local_A;
    (*B) = local_B;
}


//---------------------------------------------------------------
// (4) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO(COOElement* A, COOElement* B, const int A_counts, const int B_counts, const int n) {
    // allocation of final matrix
    double* C = calloc(n * n, sizeof(double));
    if(!C){
        fprintf(stderr, "(Multiplication) allocation error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(1);
    }

    // Multiplication: scroll A by cols and B by rows
    int i, j=0;
    for(i=0; i<A_counts && j<B_counts; i++){
        while(A[i].col>B[j].row && j<B_counts){
            j++;
        }
        int k=j;
        while(k<B_counts && A[i].col==B[k].row){
            C[n*(A[i].row) + B[k].col] += A[i].value*B[k].value;
            k++;
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
    // sorting A by cols (for reverse mode)
    if(world_rank == 0){
        start = MPI_Wtime();
        qsort(A, nnz, sizeof(COOElement), compare_col);
        end = MPI_Wtime();
        cpu_time_sorting = end - start;
    }


    //######################## 
    // matrices distribution
    //######################## 
    int A_counts;
    int B_counts;
    start = MPI_Wtime();
    comms_data_distribution(&A, &B, &A_counts, &B_counts, nnz, n, world_rank, world_size);
    end = MPI_Wtime();
    comms_time_dist = end - start;


    //######################## 
    // matrices multiplication
    //########################
    start = MPI_Wtime();
    double* C_partial = multiply_sparse_COO(A, B, A_counts, B_counts, n);
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
        printf("COO struct parallel reverse MPI (n=%d, d=%.2f)[#pr=%d]\n", n, density_percentage/100, world_size);
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
            snprintf(filename, sizeof(filename),"res/%s/output_COO_spr_MPI_%d_%d.csv", argv[1], world_size, 1);
        }
        else snprintf(filename, sizeof(filename),"res/output_COO_spr_MPI_%d_%d.csv", world_size, 1);
        FILE *fp = fopen(filename, "a");
        if (fp != NULL){
            fprintf(fp,
                "%s,%d,%.2f,%d,1,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                "COO_spr_MPI",            // algorithm
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
