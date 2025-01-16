#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int rank;
    int a=12, b=5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0){
        printf("process %d: %d + %d = %d\n", rank, a,b,a+b);
    } else if(rank==1){
        printf("process %d: %d - %d = %d\n", rank, a,b,a-b);
    } else if(rank==2){
        printf("process %d: %d * %d = %d\n", rank, a,b,a*b);
    } else if(rank==3){
        printf("process %d: %d / %d = %d\n", rank, a,b,a/b);
    } else{
        printf("process %d: %d mod %d = %d\n", rank, a,b,a%b);
    }
    
    MPI_Finalize();

    return 0;
}

