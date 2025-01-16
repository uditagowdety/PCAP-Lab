#include "mpi.h"
#include<stdio.h>
#include<string.h>

#define MAX_STRING_LEN 100

int main(int argc, char* argv[]){
    int rank,size,len;
    char str[MAX_STRING_LEN];

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        printf("enter a string in master process: ");
        fgets(str, MAX_STRING_LEN, stdin);  // Use fgets to read the entire string (including spaces)
        str[strcspn(str, "\n")] = 0;  // Remove newline character if present
        len = strlen(str) + 1;

        MPI_Send(&len,1,MPI_INT,1,0,MPI_COMM_WORLD);
        MPI_Send(str,len,MPI_CHAR,1,1,MPI_COMM_WORLD);

        // fprintf(stdout,"i have sent \"%s\" from process 0 (master process)\n",str);
        // fflush(stdout);
    } else if(rank==1) {
        MPI_Recv(&len,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
        MPI_Recv(str,len,MPI_CHAR,0,1,MPI_COMM_WORLD,&status);

        fprintf(stdout,"process %d: i have received \"%s\" in process 1 (child process) \n",rank,str);
        fflush(stdout);

        printf("toggling cases now...\n");

        for(int i=0;i<len-1;i++){
            if(str[i]>=65&&str[i]<=90){
                str[i]+=32;
            } else str[i]-=32;
        }

        MPI_Send(&len,1,MPI_INT,0,2,MPI_COMM_WORLD);
        MPI_Send(str,len,MPI_CHAR,0,3,MPI_COMM_WORLD);

        // fprintf(stdout,"i have sent toggled string from process 1 (child process)\n",str);
        // fflush(stdout);
    } else {
        printf("only 2 processes allowed for this program.");
    }

    if(rank==0){
        MPI_Recv(&len,1,MPI_INT,1,2,MPI_COMM_WORLD,&status);
        MPI_Recv(str,len,MPI_CHAR,1,3,MPI_COMM_WORLD,&status);

        fprintf(stdout, "toggled string in process 0 (master process): \"%s\"\n", str);
        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}