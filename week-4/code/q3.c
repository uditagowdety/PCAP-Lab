#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[]){
    int rank, size, fact=1, factsum=0,i;
    int n=4;
    int arr[4][4];
    int new_arr[4];

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    //in root: 
    //read 4x4 matrix

    if(rank==0){
        printf("enter your 4x4 values:\n");
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                scanf("%d",&arr[i][j]);
            }
        }

        printf("actual matrix:\n");
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                printf("%d  ",arr[i][j]);
            }
            printf("\n");
        }
    }

    //scatter each column to each process
    int row[n];
    MPI_Scatter(arr,n,MPI_INT,row,n,MPI_INT,0,MPI_COMM_WORLD);

    //mpi scan the sum and store in new matrix
    MPI_Scan(row,new_arr,n,MPI_INT,MPI_SUM, MPI_COMM_WORLD);

    MPI_Gather(new_arr,n,MPI_INT,arr,n,MPI_INT,0,MPI_COMM_WORLD);

    //in root: print the new matrix
    if(rank==0){
        printf("output matrix:\n");
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                printf("%d  ",arr[i][j]);
            }
            printf("\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}