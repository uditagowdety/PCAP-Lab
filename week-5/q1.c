#include <stdio.h>
#include <cuda.h>

#define N 1024  // Length of vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C) {
    int idx = threadIdx.x;  // Each thread handles one index
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int *h_A, *h_B, *h_C;   // Host arrays
    int *d_A, *d_B, *d_C;   // Device arrays
    int size = N * sizeof(int);

    // Allocate memory on host
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch Kernel: 1 block with N threads (Block size = N)
    vectorAdd<<<1, N>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    // Free memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
