/*
 * mpi string case toggle:
 * 
 * 1. process 0: receives input string, sends to process 1.
 * 2. process 1: receives string, toggles case (upper <-> lower), sends back to process 0.
 * 3. process 0: receives modified string, prints result.
 * 
 * uses mpi_ssend for synchronous communication.
 * designed for 2 processes (rank 0 and rank 1).
 * 
 * compile with mpicc, run with mpirun -np 2.
 */

#include "mpi.h"
#include <stdio.h>
#include <string.h>

#define MAX_STRING_LEN 100

int main(int argc, char* argv[]) {
    int rank, len;
    char str[MAX_STRING_LEN];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    if (rank == 0) {
        printf("enter string: ");
        fgets(str, MAX_STRING_LEN, stdin);
        str[strcspn(str, "\n")] = 0;
        len = strlen(str) + 1;

        MPI_Ssend(&len, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Ssend(str, len, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(str, len, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);

        for (int i = 0; i < len - 1; i++) 
            str[i] = (str[i] >= 'A' && str[i] <= 'Z') ? str[i] + 32 : (str[i] >= 'a' && str[i] <= 'z') ? str[i] - 32 : str[i];

        MPI_Ssend(&len, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Ssend(str, len, MPI_CHAR, 0, 3, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&len, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(str, len, MPI_CHAR, 1, 3, MPI_COMM_WORLD, &status);
        printf("final string in process 0: \"%s\"\n", str);
    }

    MPI_Finalize();
    return 0;
}
