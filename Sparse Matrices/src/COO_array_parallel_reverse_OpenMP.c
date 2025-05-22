//##############################################################################################################################
// Matrices Multiplication in format COO using an array as representation (OpenMP reverse version)
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
// Functions:
//---------------------------------------------------------------
//      (1) generation
//      (2) comparison (for sorting)
//      (3) multiplication
//      (4) main
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
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO_threads(int* A_row, int* A_col_offset, double* A_val, int* B_row, int* B_col, double* B_val, const int nnz, const int n, const int num_threads){
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
    for(i=0; i<=B_row[0]; i++){
        B_row_offset[i]=0;
    }
    for(i=B_row[nnz-1]+1; i<=n; i++){
        B_row_offset[i]=nnz;
    }

    // compute offsets
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<nnz; i++){
        int prev = B_row[i-1];
        int row = B_row[i];
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
                int index = A_row[i] *n +B_col[j];
                double value = A_val[i] * B_val[j];
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
    double cpu_time_computation;    //divided in sorting and multiplying
    double cpu_time_sorting;
    double cpu_time_multiplying;

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
    start = omp_get_wtime();
    generate_sparse_matrix(&A_row, &A_col, &A_val, n, nnz);
    generate_sparse_matrix(&B_row, &B_col, &B_val, n, nnz);
    end = omp_get_wtime();
    cpu_time_generation = end - start;


    //######################## 
    // sorting phase
    //########################
    start = omp_get_wtime();
    // sorting A by cols
    int* A_col_offset = sort_by_column_threads(&A_row, &A_col, &A_val, n, nnz, num_threads);
    end = omp_get_wtime();
    cpu_time_sorting = end - start;


    //######################## 
    // matrices multiplication
    //######################## 
    start = omp_get_wtime();
    double* C = multiply_sparse_COO_threads(A_row, A_col_offset, A_val, B_row, B_col, B_val, nnz, n, num_threads);
    end = omp_get_wtime(); //used also for total_time
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //######################## 
    // final section
    //########################
    //print sections time for statistics
    printf("COO array parallel reverse OpenMP (n=%d, d=%.2f)[#th=%d]\n", n, density_percentage/100, num_threads);
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
        snprintf(filename, sizeof(filename),"res/%s/output_COO_apr_OpenMP_%d_%d.csv", argv[2], 1, num_threads);
    }
    else snprintf(filename, sizeof(filename),"res/output_COO_apr_OpenMP_%d_%d.csv", 1, num_threads);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,%d,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "COO_apr_OpenMP",         // algorithm
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
    if(A_row) free(A_row);  
    if(A_col) free(A_col);
    if(A_val) free(A_val);
    if(B_row) free(B_row);  
    if(B_col) free(B_col);
    if(B_val) free(B_val);
    free(C);
    return 0;
}
