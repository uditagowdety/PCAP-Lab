#include "mpi.h"
#include<stdio.h>

int main(int argc, char* argv[]){
    int rank,size;
    char str[10];

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        printf("enter a string in master process: ");
        scanf("%s",str);
        MPI_Send(&x,1,MPI_INT,1,1,MPI_COMM_WORLD);
        fprintf(stdout,"i have sent %s from process 0 (master process)\n",str);
        fflush(stdout);
    } else {
        MPI_Recv(&x,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        fprintf(stdout,"i have received %s in process 1 (child process) \n",str);
        fprintf(stdout,"toggling cases now...");
        fflush(stdout);
    }

    if(rank==1){
        // printf("enter a string in master process: ");
        // scanf("%s",str);
        MPI_Send(&x,1,MPI_INT,0,2,MPI_COMM_WORLD);
        fprintf(stdout,"i have sent toggled string from process 1 (child process)\n",str);
        fflush(stdout);
    } else {
        MPI_Recv(&x,1,MPI_INT,1,2,MPI_COMM_WORLD,&status);
        fprintf(stdout,"toggled string in process 0 (master process): %s \n",str);
        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}