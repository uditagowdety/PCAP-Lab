#include "mpi.h"
#include <stdio.h>

int exponent(int base, int exponent);

int main(int argc, char *argv[]){
    int num,rank;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0){
        printf("enter a number: ");
        scanf("%d",&num);
    }

    MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    printf("process %d: %d^^%d = %d\n", rank,num,rank,exponent(num,rank));
    
    MPI_Finalize();

    return 0;
}

int exponent(int base, int exponent) {
    int result = 1;
    for (int i = 1; i <= exponent; i++) {
        result *= base;
    }
    return result;
}