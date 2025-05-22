//##############################################################################################################################
// Matrices Multiplication in format COO using an array as representation (sequential reverse version)
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
// Global variables
//###############################################################

// Global variable to let the compare access the array
int *A_col;



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
    if((!*row) || (!*col) || (!*value) || (!row_counts) || (!used_cols)){
        printf("Allocation error!\n");
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
// (2.1) Compare function for the qsort in function 2.2
//---------------------------------------------------------------

int compare(const void *a, const void *b) {
    int i1 = *(int *)a;
    int i2 = *(int *)b;
    return A_col[i1] - A_col[i2];
}

//---------------------------------------------------------------
// (2.2) Function for sorting
//---------------------------------------------------------------

void sort_coo(int** row, int** col, double** val, int nnz) {
    // allocation of support vector
    int* idx = malloc(nnz * sizeof(int));
    if (!idx){
        printf("malloc failed"); 
        exit(1);
    }

    // Initialize idx with {0, 1, 2, ..., n-1}
    int i;
    for(i = 0; i < nnz; i++) idx[i] = i;

    // Sort idx[] based on A_col[]
    qsort(idx, nnz, sizeof(int), compare);
 
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
    for (i = 0; i < nnz; i++) {
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
}

//---------------------------------------------------------------
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO(int* A_row, int* A_col, double* A_value, int* B_row, int* B_col, double* B_value, const int nnz, const int n) {
    // allocation of final matrix
    double* C = calloc(n*n, sizeof(double));
    if(!C){
        printf("Result allocation error!\n");
        exit(1);
    }

    // Multiplication: scroll A by cols and B by rows
    int i, j=0;
    for(i=0; i<nnz && j<nnz; i++){
        while(A_col[i]>B_row[j] && j<nnz){
            j++;
        }
        int k=j;
        while(k<nnz && A_col[i]==B_row[k]){
            C[A_row[i] *n + B_col[k]] += A_value[i]*B_value[k];
            k++;
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
    clock_t start_tot;
    start_tot = clock();

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
    int* A_row = NULL;
    A_col = NULL;
    double* A_value = NULL;
    int* B_row = NULL;
    int* B_col = NULL;
    double* B_value = NULL;


    //######################## 
    // matrices generation
    //########################
    start = clock();
    generate_sparse_matrix(&A_row, &A_col, &A_value, n, nnz);
    generate_sparse_matrix(&B_row, &B_col, &B_value, n, nnz);
    end = clock();
    cpu_time_generation = ((double) (end - start)) / CLOCKS_PER_SEC;


    //######################## 
    // sorting phase
    //########################
    start = clock();
    // sorting A by cols
    sort_coo(&A_row, &A_col, &A_value, nnz);
    end = clock();
    cpu_time_sorting = ((double) (end - start)) / CLOCKS_PER_SEC;


    //######################## 
    // matrices multiplication
    //######################## 
    start = clock();
    double* C = multiply_sparse_COO(A_row, A_col, A_value, B_row, B_col, B_value, nnz, n);
    end = clock();  //used also for total_time
    cpu_time_multiplying = ((double) (end - start)) / CLOCKS_PER_SEC;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;
    total_time = ((double) (end - start_tot)) / CLOCKS_PER_SEC;


    //######################## 
    // final section
    //########################
    //print sections time for statistics
    printf("COO array sequential (n=%d, d=%.2f)\n", n, density_percentage/100);
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
    snprintf(filename, sizeof(filename),"res/output_COO_as_%d_%d.csv", 1, 1);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,1,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "COO_a_seq",              // algorithm
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
    if(A_col) free(A_col);
    if(A_value) free(A_value);
    if(B_row) free(B_row);  
    if(B_col) free(B_col);
    if(B_value) free(B_value);
    free(C);
    return 0;
}
