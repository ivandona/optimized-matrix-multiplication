//##############################################################################################################################
// Matrices Multiplication in format COO using a struct as representation (OpenMP reverse version)
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
//      (2) sorting 
//      (3) multiplication
//      (4) main
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
        printf("Allocation error!\n");
        if(*A) free(*A);
        if(row_counts) free(row_counts);
        if(used_cols) free(used_cols);
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
        for(j = 0; j < row_counts[i]; j++){
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
int* sort_by_column_threads(COOElement** A, const int n, const int nnz, const int num_threads){
    // allocation of support vectors
    int* count = calloc(n, sizeof(int));
    int* offset = malloc((n+1) * sizeof(int));
    if((!count)||(!offset)){
        if(count) free(count);
        if(offset) free(offset);
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
    COOElement* A_final = malloc(nnz * sizeof(COOElement));
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
    free(count);
    return offset;
}


//---------------------------------------------------------------
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO_threads(COOElement* A, COOElement* B, int* A_col_offset, const int nnz, const int n, const int num_threads){
    // allocation of final matrix and support vector
    double* C = calloc(n*n, sizeof(double));
    int* B_row_offset = malloc((n+1)* sizeof(int));
    if((!C)||(!B_row_offset)){
        if(C) free(C);
        if(B_row_offset) free(B_row_offset);
        printf("Result allocation error!\n");
        exit(1);
    }

    // init first and last offsets  (very short, no need to parallelize)
    int i;
    for(i=0; i<=B[0].row; i++){
        B_row_offset[i]=0;
    }
    for(i=B[nnz-1].row+1; i<=n; i++){
        B_row_offset[i]=nnz;
    }

    // compute offsets
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<nnz; i++){
        int prev = B[i-1].row;
        int row = B[i].row;
        if(prev < row){
            while(prev < row){
                B_row_offset[++prev]=i;
            }
        }
    }

    // Multiplication: scroll A by cols and B by rows
    int k;
    for(k=0; k<n; k++){
        #pragma omp parallel for num_threads(num_threads)
        for(i=A_col_offset[k]; i<A_col_offset[k+1]; i++){
            int j;
            for(j=B_row_offset[k]; j<B_row_offset[k+1]; j++){
                int index = A[i].row *n +B[j].col;
                double value = A[i].value * B[j].value;
                C[index] += value;
            }
        }
    }
    free(B_row_offset);
    free(A_col_offset);
    return C;
}


//---------------------------------------------------------------
// (4) Main function
//---------------------------------------------------------------
int main(int argc, char** argv){
    if(argc<2){
        printf("Error, number of threads not specified");
        exit(1);
    }

    //######################## 
    // initialization
    //########################
    const int num_threads = atoi(argv[1]);

    // total time
    double start_tot;
    start_tot = omp_get_wtime();


    //######################## 
    // main variables
    //########################
    int n = 16384;  // Dimension of the matrices (n*n)
    double density_percentage = 15;  // Percentage of non null elements (es. 15 -> 15%)
    int nnz = (n * n * density_percentage/100);  // Actual non null elements
    
    double start, end;
    double cpu_time_generation;
    double cpu_time_computation;
    double cpu_time_sorting;
    double cpu_time_multiplying;

    srand(40);//(time(NULL)); //a seed is used to limit randomization to improve comparison accuracy

    // matrices variables
    COOElement* A = NULL;
    COOElement* B = NULL;


    //######################## 
    // matrices generation
    //########################
    start = omp_get_wtime();
    generate_sparse_matrix(&A, n, nnz);
    generate_sparse_matrix(&B, n, nnz);
    end = omp_get_wtime();
    cpu_time_generation = end - start;


    //######################## 
    // sorting phase
    //######################## 
    start = omp_get_wtime();
    // sorting A by cols (for reverse mode)
    int* A_col_offset = sort_by_column_threads(&A, n, nnz, num_threads);
    end = omp_get_wtime();
    cpu_time_sorting = end - start;


    //######################## 
    // matrices multiplication
    //######################## 
    start = omp_get_wtime();
    double* C = multiply_sparse_COO_threads(A, B, A_col_offset, nnz, n, num_threads);
    end = omp_get_wtime(); //used also for total_time
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //######################## 
    // final section
    //########################
    //print sections time for statistics
    printf("COO struct parallel reverse OpenMP (n=%d, d=%.2f)[#th=%d]\n", n, density_percentage/100, num_threads);
    printf("Total time: %f seconds\n", end - start_tot);
    printf("Generation time: %f seconds\n", cpu_time_generation);
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
    if(argc==3){
        snprintf(filename, sizeof(filename),"res/%s/output_COO_spr_OpenMP_%d_%d.csv", argv[2], 1, num_threads);
    }
    else snprintf(filename, sizeof(filename),"res/output_COO_spr_OpenMP_%d_%d.csv", 1, num_threads);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,%d,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "COO_spr_OpenMP",         // algorithm
            n,                        // matrix dimension
            density_percentage/100,   // matrix density
            num_threads,              // thread numbers
            end - start_tot,          // total time
            cpu_time_generation,      // generation time
            cpu_time_computation,     // total computation time
            cpu_time_sorting,         // comp. sorting time
            cpu_time_multiplying,     // comp. multiplication time
            sum                       // sum (to check correctness)
        );
        fclose(fp);
    }

    //free all memory
    free(A);
    free(B);
    free(C);
    return 0;
}
