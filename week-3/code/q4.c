// MPI Program to concatenate two strings (s1 and s2) in an alternating character-wise manner using N processes.
// The strings are divided equally among all processes. Each process alternates characters from s1 and s2 and sends the result back to the root process.
// Root process gathers the results and prints the final alternating concatenated string.

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, str_len, sub_len;
    char *s1 = NULL, *s2 = NULL, *result = NULL;
    char *sub_s1 = NULL, *sub_s2 = NULL, *sub_result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the length of the strings (divisible by %d): ", size);
        scanf("%d", &str_len);

        if (str_len % size != 0) MPI_Abort(MPI_COMM_WORLD, 1);

        s1 = (char *)malloc((str_len + 1) * sizeof(char));
        s2 = (char *)malloc((str_len + 1) * sizeof(char));

        printf("Enter string 1: ");
        scanf("%s", s1);  

        printf("Enter string 2: ");
        scanf("%s", s2);  
    }

    MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    sub_len = str_len / size;

    sub_s1 = (char *)malloc((sub_len + 1) * sizeof(char));
    sub_s2 = (char *)malloc((sub_len + 1) * sizeof(char));
    sub_result = (char *)malloc((sub_len * 2 + 1) * sizeof(char));

    MPI_Scatter(s1, sub_len, MPI_CHAR, sub_s1, sub_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, sub_len, MPI_CHAR, sub_s2, sub_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < sub_len; i++) {
        sub_result[2 * i] = sub_s1[i]; 
        sub_result[2 * i + 1] = sub_s2[i]; 
    }
    sub_result[2 * sub_len] = '\0'; 

    if (rank == 0) result = (char *)malloc((str_len * 2 + 1) * sizeof(char)); 

    MPI_Gather(sub_result, sub_len * 2, MPI_CHAR, result, sub_len * 2, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultant string: %s\n", result);
        free(s1); free(s2); free(result);
    }

    free(sub_s1); free(sub_s2); free(sub_result);

    MPI_Finalize();
    return 0;
}
