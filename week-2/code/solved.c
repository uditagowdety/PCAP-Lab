#include "mpi.h"
#include<stdio.h>

int main(int argc, char* argv[]){
    int rank,size,x;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        printf("enter a value in master process: ");
        scanf("%d",&x);
        MPI_Send(&x,1,MPI_INT,1,1,MPI_COMM_WORLD);
        fprintf(stdout,"i have sent %d from process 0 (master process)\n",x);
        fflush(stdout);
    } else {
        MPI_Recv(&x,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        fprintf(stdout,"i have received %d in process 1 (child process) \n",x);
        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}