#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void modifymatrix(int* matrix, int* outputmatrix, int m, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < m && col < n) {
        if (row == 0 || row == m - 1 || col == 0 || col == n - 1) {
            outputmatrix[row * n + col] = matrix[row * n + col];
        } else {
            int value = matrix[row * n + col];
            outputmatrix[row * n + col] = ~value; 
        }
    }
}

int main() {
    int m, n;
    printf("Enter matrix dimensions (m x n): ");
    scanf("%d %d", &m, &n);

    int* matrix = (int*)malloc(m * n * sizeof(int));
    int* outputmatrix = (int*)malloc(m * n * sizeof(int));

    printf("Enter matrix elements:\n");
    for (int i = 0; i < m * n; i++) {
        scanf("%d", &matrix[i]);
    }

    int* d_matrix, *d_outputmatrix;
    cudaMalloc((void**)&d_matrix, m * n * sizeof(int));
    cudaMalloc((void**)&d_outputmatrix, m * n * sizeof(int));

    cudaMemcpy(d_matrix, matrix, m * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    modifymatrix<<<numBlocks, threadsPerBlock>>>(d_matrix, d_outputmatrix, m, n);

    cudaMemcpy(outputmatrix, d_outputmatrix, m * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nModified Matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", outputmatrix[i * n + j]);
        }
        printf("\n");
    }

    free(matrix);
    free(outputmatrix);
    cudaFree(d_matrix);
    cudaFree(d_outputmatrix);

    return 0;
}
