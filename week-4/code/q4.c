#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    char str[100];            // Input string from rank 0
    char local_str[100];      // Local buffer for each process
    char final_str[200];      // Final string at rank 0 (sufficient size)
    int str_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
        printf("Enter a word (length=number of processes): ");
        scanf("%s", str);
        str_len = strlen(str);

        // Send each character to other processes
        for (int i = 1; i < size; i++) {
            MPI_Send(&str[i], 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        // Rank 0 appends its character 1 time
        local_str[0] = str[0];
    } else {
        // Non-root processes receive their respective character
        MPI_Recv(local_str, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Each process appends its character (rank + 1 times)
    int count = rank + 1;  // Rank 0 appends 1 time, rank 1 appends 2 times, etc.
    for (int i = 1; i < count; i++) {
        local_str[i] = str[rank];  // Fill local_str with its character
    }

    // Now we gather the data into the final string at rank 0 using MPI_Gatherv
    int *send_counts = NULL;
    int *displacements = NULL;

    if (rank == 0) {
        // Allocate and initialize send_counts and displacements
        send_counts = (int*)malloc(size * sizeof(int));
        displacements = (int*)malloc(size * sizeof(int));

        // Calculate the number of characters each process will send and the displacement
        int displacement = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = i + 1;  // Each process sends (rank + 1) characters
            displacements[i] = displacement;
            displacement += i + 1;  // Update displacement for the next process
        }
    }

    // Gather the local strings into final_str at rank 0
    MPI_Gatherv(local_str, count, MPI_CHAR, final_str, send_counts, displacements, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Rank 0: Print the final string
    if (rank == 0) {
        final_str[str_len] = '\0';  // Null terminate the final string
        printf("Resulting string: %s\n", final_str);
        free(send_counts);  // Free dynamically allocated memory
        free(displacements);  // Free dynamically allocated memory
    }

    MPI_Finalize();
    return 0;
}
