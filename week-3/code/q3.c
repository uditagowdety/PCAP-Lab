// MPI program to count the number of consonants in a string
// The string is divided among N processes (including the root)
// The root process receives the total count of consonants from all processes

#include "mpi.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

int count_consonants(char* str, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        char c = tolower(str[i]);
        if ((c >= 'a' && c <= 'z') && !(c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')) {
            count++;
        }
    }
    return count;
}

int main(int argc, char* argv[]) {
    int rank, size, str_len, sub_len, local_consonant_count = 0, total_consonant_count = 0;
    char *str = NULL, *sub_str = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        str = (char*)malloc(1000);
        printf("Enter a string: ");
        fgets(str, 1000, stdin);
        str_len = strlen(str);
        if (str[str_len - 1] == '\n') str[--str_len] = '\0';
        if (str_len % size != 0) {
            printf("Error: String length must be divisible by process count.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        sub_len = str_len / size;
    }

    MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sub_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    sub_str = (char*)malloc(sub_len + 1);
    MPI_Scatter(str, sub_len, MPI_CHAR, sub_str, sub_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_consonant_count = count_consonants(sub_str, sub_len);
    MPI_Reduce(&local_consonant_count, &total_consonant_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total consonants: %d\n", total_consonant_count);
        free(str);
    }

    free(sub_str);
    MPI_Finalize();
    return 0;
}
