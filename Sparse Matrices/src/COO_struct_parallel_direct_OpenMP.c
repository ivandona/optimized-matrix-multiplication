//##############################################################################################################################
// Matrices Multiplication in format COO using a struct as representation (OpenMP direct version)
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
int* sort_by_row_col_threads(COOElement* A, const int n, const int nnz, const int num_threads){
    int* A_offset = malloc((n+1) * sizeof(int));
    if(!A_offset){
        printf("Sorting result allocation error!\n");
        exit(1);
    }

    // init first and last offsets  (very short, no need to parallelize)
    int i;
    for(i=0; i<=A[0].row; i++){
        A_offset[i]=0;
    }
    for(i=A[nnz-1].row+1; i<=n; i++){
        A_offset[i]=nnz;
    }

    // compute offsets
    #pragma omp parallel for num_threads(num_threads)
    for(i=1; i<nnz; i++){
        int prev = A[i-1].row;
        int row = A[i].row;
        if(prev < row){
            while(prev < row){
                A_offset[++prev]=i;
            }
        }
    }

    // sort all the sub array (A is already ordered by rows)
    #pragma omp parallel for num_threads(num_threads)
    for(i=0; i<n; i++){
        int index = A_offset[i];
        int count = A_offset[i+1] - index;
        if(count>0){
            qsort(&(A[index]), count, sizeof(COOElement), compare_col);
        }
    }
    return A_offset;
}


//---------------------------------------------------------------
// (3) Function for multiplication
//---------------------------------------------------------------
double* multiply_sparse_COO_threads(COOElement* A, COOElement* B, int* A_row_offset, int* B_col_offset, const int n, const int num_threads){
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
    //sorting B by cols and A by rows and cols
    int* B_col_offset = sort_by_column_threads(&B, n, nnz, num_threads);
    int* A_row_offset = sort_by_row_col_threads(A, n, nnz, num_threads);
    end = omp_get_wtime();
    cpu_time_sorting = end - start;

    
    //######################## 
    // matrices multiplication
    //######################## 
    start = omp_get_wtime();
    double* C = multiply_sparse_COO_threads(A, B, A_row_offset, B_col_offset, n, num_threads);
    end = omp_get_wtime(); //used also for total_time
    cpu_time_multiplying = end - start;
    cpu_time_computation = cpu_time_sorting + cpu_time_multiplying;


    //######################## 
    // final section
    //########################
    //print sections time for statistics
    printf("COO struct parallel direct OpenMP (n=%d, d=%.2f)[#th=%d]\n", n, density_percentage/100, num_threads);
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
        snprintf(filename, sizeof(filename),"res/%s/output_COO_spd_OpenMP_%d_%d.csv", argv[2], 1, num_threads);
    }
    else snprintf(filename, sizeof(filename),"res/output_COO_spd_OpenMP_%d_%d.csv", 1, num_threads);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,%.2f,1,%d,%.3f,%.3f,0,0,0,%.3f,%.3f,%.3f,%.3f\n",
            "COO_spd_OpenMP",         // algorithm
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
