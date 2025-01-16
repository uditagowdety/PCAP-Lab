#include "mpi.h"
#include<stdio.h>

int squared(int x){
    return x*x;
}

int cubed(int x){
    return x*x*x;
}

int main(int argc, char* argv[]){
    int rank,size;
    int arr[10]={1,2,3,4,5,6,7,8,9,10};

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        for(int i=1;i<size;i++){
            MPI_Send(&arr[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&arr[rank],1,MPI_INT,0,0,MPI_COMM_WORLD,&status);

        if(rank%2==0){
            printf("process %d: received %d, squared result: %d\n",rank,arr[rank],squared(arr[rank]));
        } else {
            printf("process %d: received %d, cubed result: %d\n",rank,arr[rank],cubed(arr[rank]));
        }
    }

    MPI_Finalize();

    return 0;
}