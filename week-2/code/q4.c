#include "mpi.h"
#include<stdio.h>

int main(int argc, char* argv[]){
    int rank,size,x;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        //receive input
        printf("enter input number: ");
        scanf("%d",&x);

        //increment
        x++;

        //send to rank 1
        MPI_Ssend(&x,1,MPI_INT,1,0,MPI_COMM_WORLD);

        fprintf(stdout,"process %d: sending %d to process %d\n",rank,x,rank+1);
        fflush(stdout);
    } else {
        // mpi_recv(from i-1)
        MPI_Recv(&x,1,MPI_INT,rank-1,0,MPI_COMM_WORLD,&status);
        
        // increment by 1
        x++;
        
        // mpi_send(to i+1)
        MPI_Ssend(&x,1,MPI_INT,(rank+1)%size,0,MPI_COMM_WORLD);

        fprintf(stdout,"process %d: sending %d to process %d\n",rank,x,rank+1);
        fflush(stdout);
    }

    if(rank==0){
        MPI_Recv(&x, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);
        fprintf(stdout,"back at process %d, final value: %d\n",rank,x);
        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}