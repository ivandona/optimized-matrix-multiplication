//##############################################################################################################################
// Matrices Multiplication in format CSR using an array as representation (sequential reverse version)
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
        exit(1);
    }

    int i;
    for(i=0; i<nnz; i++){
        col_counts[col_csr[i]]++;
    }

    col_ptr_csc[0] = 0;
    for(i=0; i<n; i++){
        col_ptr_csc[i+1] = col_ptr_csc[i] + col_counts[i];
    }

    int j;
    for(i=0; i<n; i++){
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
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_CSC_x_CRS(int* A_row, int* A_col_ptr, double* A_value, int* B_row_ptr, int* B_col, double* B_value, const int nnz, const int n){
    // allocation of final matrix
    double* C = calloc(n*n, sizeof(double));
    if(!C){
        printf("Result allocation error!\n");
        exit(1);
    }

    // Multiplication: scroll A by cols and B by rows
    int k, i, j;
    for(k=0; k<n; k++){
        for(i=A_col_ptr[k]; i<A_col_ptr[k+1]; i++){
            for(j=B_row_ptr[k]; j<B_row_ptr[k+1]; j++){
                C[A_row[i]*n + B_col[j]] += A_value[i] * B_value[j];
            }
        }
    }
    return C;
}


//---------------------------------------------------------------
// (4) Main function
//---------------------------------------------------------------
int main(){
    //########################
    // main variables
    //########################
    double total_time;
    clock_t start_tot = clock();

    int n = 16384;  // Dimension of the matrices (n*n)
    double density_percentage = 15;  // Percentage of non null elements (es. 15 -> 15%)
    int nnz = (n * n * density_percentage/100);  // Actual non null elements

    clock_t start, end;
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
    start = clock();
    generate_sparse_matrix(&A_row_ptr, &A_col, &A_value, n, nnz);
    generate_sparse_matrix(&B_row_ptr, &B_col, &B_value, n, nnz);
    end = clock();
    cpu_time_generation = ((double) (end - start)) / CLOCKS_PER_SEC;


    //########################
    // sorting phase
    //########################
    start = clock();
    // converting A from CSR to CSC
    int* A_col_ptr = NULL;
    int* A_row = NULL;
    convert_CSR_to_CSC(A_row_ptr, A_col, &A_value, &A_col_ptr, &A_row, n, nnz);
    end = clock();
    cpu_time_sorting = ((double) (end - start)) / CLOCKS_PER_SEC;


    //########################
    // matrices multiplication
    //########################
    start = clock();
    double* C = multiply_sparse_CSC_x_CRS(A_row, A_col_ptr, A_value, B_row_ptr, B_col, B_value, nnz, n);
    end = clock();  //used also for total_time
    cpu_time_multiplying = ((double) (end - start)) / CLOCKS_PER_SEC;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;
    total_time = ((double) (end - start_tot)) / CLOCKS_PER_SEC;


    //########################
    // final section
    //########################
    //print sections time for statistics
    printf("CSR array sequential (n=%d, d=%.2f)\n", n, density_percentage/100);
    printf("Total time: %.2f seconds\n", total_time);
    printf("Generation time: %.2f seconds\n", cpu_time_generation);
    printf("Computation time (total): %.2f seconds\n", cpu_time_computation);
    printf("Computation time (sorting): %.2f seconds\n", cpu_time_sorting);
    printf("Computation time (multiplication): %.2f seconds\n", cpu_time_multiplying);

    // computing the sum of all elements to check code correctness (require fix seed for rand())
    double sum=0;
    int i;
    int j;
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            sum+=C[i*n+j];
        }
    }
    printf("Final C matrix, elements sum = %lf\n",sum);
    char filename[128];
    snprintf(filename, sizeof(filename),"res/output_CSR_as_%d_%d.csv", 1, 1);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,1,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "CSR_a_seq",              // algorithm
            n,                        // matrix dimension
            density_percentage/100,   // matrix density
            total_time,               // total time
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
    if(A_col_ptr) free(A_col_ptr);
    if(A_value) free(A_value);
    if(B_row_ptr) free(B_row_ptr);
    if(B_col) free(B_col);
    if(B_value) free(B_value);
    free(C);
    return 0;
}
