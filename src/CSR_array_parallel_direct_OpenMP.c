//##############################################################################################################################
// Matrices Multiplication in format CSR using an array as representation (OpenMP direct version)
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
//      (2) sorting (conversion)
//      (3) multiplication
//      (4) main
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
void convert_CSR_to_CSC_threads(int* row_ptr_csr, int* col_csr, double** val, int** col_ptr, int** row, const int n, const int nnz, const int num_threads){
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
        exit(1);
    }

    int i;
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<nnz; i++){
        int col = col_csr[i];
        #pragma omp atomic
        col_counts[col]++;
    }

    col_ptr_csc[0] = 0;
    for(i=0; i<n; i++){ //better to keep it serial (n is relatively small)
        col_ptr_csc[i+1] = col_ptr_csc[i] + col_counts[i];
    }

    for(i=n-1; i>=0; i--){
        int j;
        #pragma omp parallel for num_threads(num_threads)
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
void sorting_CSR_threads(int* row_ptr, int** col, double** val, const int n, const int nnz, const int num_threads){
    int i;
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<n; i++){
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        int* idx = calloc(n, sizeof(int));
        if(!idx){
            printf("Sorting allocation error!\n");
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
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_CSC_x_CRS_threads(int* A_row_ptr, int* A_col, double* A_value, int* B_row, int* B_col_ptr, double* B_value, const int n, const int num_threads){
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
    double start_tot = omp_get_wtime();


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
    int* A_row_ptr = NULL;
    int* A_col = NULL;
    double* A_value = NULL;
    int* B_row_ptr = NULL;
    int* B_col = NULL;
    double* B_value = NULL;


    //########################
    // matrices generation
    //########################
    start = omp_get_wtime();
    generate_sparse_matrix(&A_row_ptr, &A_col, &A_value, n, nnz);
    generate_sparse_matrix(&B_row_ptr, &B_col, &B_value, n, nnz);
    end = omp_get_wtime();
    cpu_time_generation = end - start;


    //########################
    // sorting phase
    //########################
    // converting B from CSR to CSC
    int* B_col_ptr = NULL;
    int* B_row = NULL;
    start = omp_get_wtime();
    convert_CSR_to_CSC_threads(B_row_ptr, B_col, &B_value, &B_col_ptr, &B_row, n, nnz, num_threads);
    // sorting A by row and col
    sorting_CSR_threads(A_row_ptr, &A_col, &A_value, n, nnz, num_threads);
    end = omp_get_wtime();
    cpu_time_sorting = end - start;
    

    //########################
    // matrices multiplication
    //########################
    start = omp_get_wtime();
    double* C = multiply_sparse_CSC_x_CRS_threads(A_row_ptr, A_col, A_value, B_row, B_col_ptr, B_value, n, num_threads);
    end = omp_get_wtime(); //used also for total_time
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //########################
    // final section
    //########################
    //print sections time for statistics
    printf("CSR array parallel direct OpenMP (n=%d, d=%.2f)[#th=%d]\n", n, density_percentage/100, num_threads);
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
        snprintf(filename, sizeof(filename),"res/%s/output_CSR_apd_OpenMP_%d_%d.csv", argv[2], 1, num_threads);
    }
    else snprintf(filename, sizeof(filename),"res/output_CSR_apd_OpenMP_%d_%d.csv", 1, num_threads);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,%d,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "CSR_apd_OpenMP",         // algorithm
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
    free(A_row_ptr);
    free(A_col);
    free(A_value);
    free(B_row);
    free(B_col_ptr);
    free(B_value);
    free(C);
    return 0;
}
