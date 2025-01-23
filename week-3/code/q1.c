#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int factorial(int n) {
    int fact = 1;
    for (int i = 1; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char* argv[]) {
    int rank, size, x, sum_of_factorials = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int* numbers = (int*)malloc(sizeof(int) * (size - 1));
        for (int i = 1; i < size; i++) {
            printf("Enter value for process %d: ", i);
            scanf("%d", &numbers[i - 1]);
        }
        MPI_Scatter(numbers, 1, MPI_INT, &x, 1, MPI_INT, 0, MPI_COMM_WORLD);
        free(numbers);
    } else {
        MPI_Scatter(NULL, 1, MPI_INT, &x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int fact_result = factorial(x);
    MPI_Reduce(&fact_result, &sum_of_factorials, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum of factorials: %d\n", sum_of_factorials);
    }

    MPI_Finalize();
    return 0;
}
