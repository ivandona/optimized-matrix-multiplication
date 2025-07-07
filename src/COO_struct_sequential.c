//##############################################################################################################################
// Matrices Multiplication in format COO using a struct as representation (sequential reverse version)
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
int compare_col(const void* a, const void* b) {
    return ((COOElement*)a)->col - ((COOElement*)b)->col;
}


//---------------------------------------------------------------
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO(COOElement* A, COOElement* B, const int nnz, const int n){
    // allocation of final matrix
    double* C = calloc(n*n, sizeof(double));
    if(!C){
        printf("Result allocation error!\n");
        exit(1);
    }

    // Multiplication: scroll A by cols and B by rows
    int i, j=0;
    for(i=0; i<nnz && j<nnz; i++){
        while(A[i].col>B[j].row && j<nnz){
            j++;
        }
        int k=j;
        while(k<nnz && A[i].col==B[k].row){
            C[A[i].row *n + B[k].col] += A[i].value*B[k].value;
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
    COOElement* A = NULL;
    COOElement* B = NULL;


    //######################## 
    // matrices generation
    //########################
    start = clock();
    generate_sparse_matrix(&A, n, nnz);
    generate_sparse_matrix(&B, n, nnz);
    end = clock();
    cpu_time_generation = ((double) (end - start)) / CLOCKS_PER_SEC;



    //######################## 
    // sorting phase
    //########################
    start = clock();
    // sorting A by cols (for reverse mode)
    qsort(A, nnz, sizeof(COOElement), compare_col);
    end = clock();
    cpu_time_sorting = ((double) (end - start)) / CLOCKS_PER_SEC;


    //######################## 
    // matrices multiplication
    //######################## 
    start = clock();
    double* C = multiply_sparse_COO(A, B, nnz, n);
    end = clock();  //used also for total_time
    cpu_time_multiplying = ((double) (end - start)) / CLOCKS_PER_SEC;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;
    total_time = ((double) (end - start_tot)) / CLOCKS_PER_SEC;


    //######################## 
    // final section
    //########################
    //print sections time for statistics
    printf("COO struct sequential (n=%d, d=%.2f)\n", n, density_percentage/100);
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
    snprintf(filename, sizeof(filename),"res/output_COO_ss_%d_%d.csv", 1, 1);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,1,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "COO_s_seq",              // algorithm
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
    free(A);
    free(B);
    free(C);
    return 0;
}
