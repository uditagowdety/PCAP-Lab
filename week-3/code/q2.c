// MPI program to calculate the average of averages:
// 1. Root process (rank 0) reads a matrix of size m x n, where n is the number of processes.
// 2. The matrix is divided row-wise and scattered to all processes.
// 3. Each process calculates the average of its assigned rows.
// 4. Root process gathers the local averages and computes the final average of all averages.

#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[]){
    int rank, size, m, *numbers = NULL, *subarray = NULL;
    float *averages = NULL, local_avg, root_avg = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        printf("enter value of m: ");
        scanf("%d", &m);
        numbers = (int*)malloc(m * size * sizeof(int));
        printf("enter mxn values:\n");
        for(int i = 0; i < m * size; i++) scanf("%d", &numbers[i]);
        averages = (float*)malloc(size * sizeof(float));
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    subarray = (int*)malloc(m * sizeof(int));

    MPI_Scatter(numbers, m, MPI_INT, subarray, m, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for(int i = 0; i < m; i++) local_sum += subarray[i];
    local_avg = (float)local_sum / m;

    MPI_Gather(&local_avg, 1, MPI_FLOAT, averages, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        for(int i = 0; i < size; i++) root_avg += averages[i];
        root_avg /= size;
        printf("final average of all averages: %f\n", root_avg);
        free(numbers);
        free(averages);
    }

    free(subarray);
    MPI_Finalize();
    return 0;
}
