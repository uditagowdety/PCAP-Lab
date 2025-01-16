#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int rank;
    char str[]="UdiTA";

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(str[rank]>=65 && str[rank]<=90){
        str[rank]+=32;
    } else{
        str[rank]-=32;
    }

    printf("process %d: %s\n",rank, str);

    MPI_Finalize();

    return 0;
}

