/*
 * mpi ring pass:
 * 1. process 0 receives input value, increments by 1, and sends to process 1.
 * 2. other processes (1 to n-1) receive value from previous process, increment by 1, send to next process.
 * 3. process 0 receives final incremented value and prints it.
 * uses mpi_ssend for synchronous send.
 * designed for n processes in a ring.
 * compile with mpicc, run with mpirun -np n.
 */

#include "mpi.h"
#include<stdio.h>

int main(int argc, char* argv[]){
    int rank, size, x;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank == 0){
        printf("enter input number: ");
        scanf("%d", &x);
        x++; // increment
        MPI_Ssend(&x, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&x, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        x++; // increment
        MPI_Ssend(&x, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        MPI_Recv(&x, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);
        printf("back at process %d, final value: %d\n", rank, x);
    }

    MPI_Finalize();
    return 0;
}
