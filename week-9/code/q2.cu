#include <stdio.h>
#include <cuda_runtime.h>

__global__ void modifyMatrix(int* matrix, int m, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < m && col < n) {
        int power = row + 1;  

        int value = matrix[row * n + col];
        int result = 1;
        
        for (int i = 0; i < power; i++) {
            result *= value;
        }

        matrix[row * n + col] = result;
    }
}


int main() {
    int m, n;
    printf("enter matrix dimensions (m x n): ");
    scanf("%d %d", &m, &n);

    int *matrix = (int*)malloc(m * n * sizeof(int));

    printf("enter matrix elements:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &matrix[i * n + j]);
        }
    }

    int *d_matrix;
    cudaMalloc((void**)&d_matrix, m * n * sizeof(int));

    cudaMemcpy(d_matrix, matrix, m * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + 15) / 16, (n + 15) / 16);

    modifyMatrix<<<numBlocks, threadsPerBlock>>>(d_matrix, m, n);

    cudaMemcpy(matrix, d_matrix, m * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nmodified matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }

    free(matrix);
    cudaFree(d_matrix);

    return 0;
}
