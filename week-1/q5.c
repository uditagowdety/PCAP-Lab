#include "mpi.h"
#include <stdio.h>

int factorial(int num);
int fibonacci(int n);

int main(int argc, char *argv[]){
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank%2==0){
        printf("process %d: fact(%d) = %d\n", rank, rank,factorial(rank));
    } else{
        printf("process %d: fibonacci(%d) = %d\n", rank, rank,fibonacci(rank));
    }
    
    MPI_Finalize();

    return 0;
}

int factorial(int num){
    int fact=1;

    while(num>0){
        fact*=num;
        num--;
    }

    return fact;
}

int fibonacci(int n){
    if(n==0) return 0;
    if(n==1) return 1;

    return fibonacci(n-1)+fibonacci(n-2);
}
