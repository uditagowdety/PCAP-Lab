//todo: understand what you're sending as an arg to search_count
//todo: check out why total_sum isn't working right

#include "mpi.h"
#include <stdio.h>

int search_count(int num, int arr[], int n){
    int count=0;

    for(int i=0;i<n;i++){
        if(arr[i]==num) count++;
    }

    return count;
}

int main(int argc, char* argv[]){
    int rank, size, search_num,sub_count,total_count=0, arr[3][3];

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    //read 3x3 matrix
    if(rank==0){
        printf("enter values of 3x3 matrix:\n");
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                scanf("%d",&arr[i][j]);
            }
        }

        //in root process: enter element to search for in the matrix

        printf("enter number to be searched: ");
        scanf("%d",&search_num);
        
    }

    MPI_Bcast(&search_num,1,MPI_INT,0,MPI_COMM_WORLD); 

    int row[3];
    MPI_Scatter(arr,3,MPI_INT, row,3,MPI_INT,0,MPI_COMM_WORLD);

    // printf("process %d: searching for %d\n",rank, search_num); ((SO BCAST WORKS))

    //use three processes --> find no. of occurences (maybe using a helper function?)
    sub_count=search_count(search_num, row,3);
    printf("process %d: subcount = %d\n",rank, sub_count);
    
    MPI_Reduce(&sub_count,&total_count,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    if (rank==0){
        printf("number of times %d occurred: %d\n",search_num,total_count);
    }

    MPI_Finalize();
    return 0;
}