//todo: fix the scan nonsense, factsum isnt taking all sums for some reason
//todo: add some errors that can be handled by routines

#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[]){
    int rank, size, fact=1, factsum=0,i;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    for(int i=1;i<=rank+1;i++){
        fact*=i;
    }

    printf("process %d: fact = %d\n",rank,fact);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scan(&fact,&factsum,1,MPI_INT,MPI_SUM, MPI_COMM_WORLD);

    printf("Process %d: factsum = %d\n", rank, factsum);

    if(rank==size-1){
        printf("sum of all factorials: %d\n",factsum);
    }

    MPI_Finalize();
    return 0;
}