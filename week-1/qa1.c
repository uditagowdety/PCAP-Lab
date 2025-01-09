#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int rank;
    int nums[]={18,523,301,1234,2,14,108,150,1928};
    int newarr[9];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int current=nums[rank];
    int new=0;

    while(current>0){
        int digit=current%10;
        new=new*10+digit;
        current/=10;
    }

    nums[rank]=new;
    newarr[rank]=new;

    printf("process %d: %d\n",rank,nums[rank]);
    
    MPI_Finalize();
    
    return 0;
}

