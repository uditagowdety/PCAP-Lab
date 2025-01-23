#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[]){
    int rank,size,m,n;
    int* numbers=NULL;
    int* subarray=NULL;
    float* averages=NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    n=size;

    if(rank==0){
        printf("enter value of m: ");
        scanf("%d",&m);

        numbers=(int*)malloc(m*n*sizeof(int));

        printf("enter mxn values:\n");
        for(int i=0;i<m*n;i++){
            scanf("%d",&numbers[i]);
        }

        averages=(float*)malloc(n*sizeof(float));
    } 
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    subarray=(int*)malloc(m*sizeof(int));

    // printf("Rank %d: m=%d, n=%d, sendcount=%d, recvcount=%d\n", rank, m, n, m, m);

    MPI_Scatter(numbers,m,MPI_INT,subarray,m,MPI_INT,0,MPI_COMM_WORLD);

    int local_sum=0;
    for(int i=0;i<m;i++){
        local_sum+=subarray[i];
    }
    float local_avg=(float)local_sum/m;

    printf("process %d: average = %f\n",rank,local_avg);

    MPI_Gather(&local_avg,1,MPI_FLOAT,averages,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    if(rank==0){
        float root_avg=0;
        for(int i=0;i<n;i++){
            root_avg+=averages[i];
        }
        root_avg/=n;

        printf("final average of all averages: %f\n",root_avg);

        free(numbers);
        free(averages);
    }

    free(subarray);

    MPI_Finalize();

    return 0;
}

//read an integer value M, and N is the number of processes
//read NxM elements into a 1D array in root process
//root process sends M elements to each process
//each process finds average of M elements that it received. sends the avg back to root
//root collects all averages and finds average of them

//notes:
// Plan:

//     Input Handling (Root Process):
//         The root process (rank 0) will read the value of M and N.
//         The root will read NxM elements into a 1D array.

//     Distribute Data (MPI_Scatter):
//         Use MPI_Scatter to send M elements to each of the processes, including the root process.

//     Average Calculation (All Processes):
//         Each process will calculate the average of the M elements it received.

//     Send Back Averages to Root (MPI_Gather):
//         Each process sends its calculated average back to the root using MPI_Gather.

//     Root Computes Final Average:
//         The root process will gather all averages and compute the final average of these averages.