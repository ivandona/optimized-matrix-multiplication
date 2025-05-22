//##############################################################################################################################
// Matrices Multiplication in format COO using an array as representation (OpenMP direct version)
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
int* sort_by_row_col_threads(int** row, int** col, double** val, const int n, const int nnz, const int num_threads){
    int* offset = malloc((n+1) * sizeof(int));
    if(!offset){
        printf("Sorting result allocation error!\n");
        exit(1);
    }

    // init first and last offsets  (very short, no need to parallelize)
    int i;
    for(i=0; i<=(*row)[0]; i++){
        offset[i]=0;
    }
    for(i=(*row)[nnz-1]+1; i<=n; i++){
        offset[i]=nnz;
    }

    // compute offsets
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<nnz; i++){
        int prev = (*row)[i-1];
        int r = (*row)[i];
        if(prev < r){
            while(prev < r){
                offset[++prev]=i;
            }
        }
    }

    // sort all the sub array (A is already ordered by rows)
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<n; i++){
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
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO_threads(int* A_row_offset, int* A_col, double* A_val, int* B_row, int* B_col_offset, double* B_val, const int n, const int num_threads){
    // allocation of final matrix
    double* C = calloc(n*n, sizeof(double));
    if(!C){
        printf("Result allocation error!\n");
        exit(1);
    }

    // Multiplication: scroll A by rows and B by cols
    int A_row;
    #pragma omp parallel for num_threads(num_threads)
    for(A_row = 0; A_row<n; A_row++){
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
    //sorting B by cols and A by rows and cols
    int* A_row_offset = sort_by_row_col_threads(&A_row, &A_col, &A_val, n, nnz, num_threads);
    int* B_col_offset = sort_by_column_threads(&B_row, &B_col, &B_val, n, nnz, num_threads);
    end = omp_get_wtime();
    cpu_time_sorting = end - start;


    //######################## 
    // matrices multiplication
    //######################## 
    start = omp_get_wtime();
    double* C = multiply_sparse_COO_threads(A_row_offset, A_col, A_val, B_row, B_col_offset, B_val, n, num_threads);
    end = omp_get_wtime(); //used also for total_time
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //######################## 
    // final section
    //########################
    //print sections time for statistics
    printf("COO array parallel direct OpenMP (n=%d, d=%.2f)[#th=%d]\n", n, density_percentage/100, num_threads);
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
        snprintf(filename, sizeof(filename),"res/%s/output_COO_apd_OpenMP_%d_%d.csv", argv[2], 1, num_threads);
    }
    else snprintf(filename, sizeof(filename),"res/output_COO_apd_OpenMP_%d_%d.csv", 1, num_threads);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,%d,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "COO_apd_OpenMP",         // algorithm
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
