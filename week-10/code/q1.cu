#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul(int* A, int* B, int* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main() {
    int m, n, p;
    printf("enter matrix dimensions (m x n) and (n x p) for A and B: ");
    scanf("%d %d %d", &m, &n, &p);

    int* A = (int*)malloc(m * n * sizeof(int));
    int* B = (int*)malloc(n * p * sizeof(int));
    int* C = (int*)malloc(m * p * sizeof(int));

    printf("enter matrix A:\n");
    for (int i = 0; i < m * n; i++) {
        scanf("%d", &A[i]);
    }

    printf("enter matrix B:\n");
    for (int i = 0; i < n * p; i++) {
        scanf("%d", &B[i]);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(int));
    cudaMalloc((void**)&d_B, n * p * sizeof(int));
    cudaMalloc((void**)&d_C, m * p * sizeof(int));

    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C, d_C, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResulting matrix C:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%d ", C[i * p + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
