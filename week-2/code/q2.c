/*
 * mpi send value from master to all children
 * 
 * 1. process 0 asks for input and sends to all processes
 * 2. each child receives and prints value with rank
 * 
 * uses mpi_send and mpi_recv
 * 
 * compile: mpicc program.c -o program
 * run: mpirun -np N ./program
 * (N = 1 master + N-1 children)
 */

#include "mpi.h"
#include<stdio.h>

int main(int argc, char* argv[]){
    int rank, size, x;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(rank==0){
        printf("enter a value: ");
        scanf("%d", &x);
        for(int i=1; i<size; i++) MPI_Send(&x, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        printf("sent %d to all child processes\n", x);
    } else {
        MPI_Recv(&x, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("received %d in process %d\n", x, rank);
    }

    MPI_Finalize();
    return 0;
}