/*
 * mpi program using buffered send:
 * 1. process 0 sends elements of an array (arr) to all other processes (1 to size-1).
 * 2. each child process receives the value and either squares (if rank is even) or cubes (if rank is odd) the value.
 * 3. uses MPI_Bsend for buffered communication, which requires buffer attachment before communication.
 * 4. after communication, MPI_Buffer_detach is used to release the buffer.
 * 5. designed for 1 master process (rank 0) and multiple child processes (rank 1 to size-1).
 * 
 * compile with mpicc, run with mpirun -np <num_processes> to execute.
 */

#include "mpi.h"
#include<stdio.h>

int squared(int x) { return x * x; }
int cubed(int x) { return x * x * x; }

int main(int argc, char* argv[]) {
    int rank, size;
    int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    int buffer[100];
    MPI_Buffer_attach(buffer, sizeof(buffer));

    if (rank == 0) {
        for (int i = 1; i < size; i++) 
            MPI_Bsend(&arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&arr[rank], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("process %d: received %d, result: %d\n", rank, arr[rank], (rank % 2 == 0) ? squared(arr[rank]) : cubed(arr[rank]));
    }

    MPI_Buffer_detach(&buffer, NULL);
    MPI_Finalize();
    return 0;
}
