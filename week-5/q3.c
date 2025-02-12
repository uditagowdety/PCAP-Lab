#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define N 1024  // Number of elements
#define THREADS_PER_BLOCK 256  // Fixed number of threads per block

// CUDA Kernel for computing sine of each angle
__global__ void computeSine(float *input, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Compute global index
    if (idx < n) {
        output[idx] = sinf(input[idx]);  // Compute sine using CUDA's sinf()
    }
}

int main() {
    float *h_input, *h_output;   // Host arrays
    float *d_input, *d_output;   // Device arrays
    int size = N * sizeof(float);

    // Allocate memory on host
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    // Initialize input array with angles in radians
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 0.01f;  // Example: Generating angles (0, 0.01, 0.02, ...)
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Calculate number of blocks dynamically
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch Kernel: 'blocks' blocks, 'THREADS_PER_BLOCK' threads per block
    computeSine<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("sin(%.2f) = %.6f\n", h_input[i], h_output[i]);
    }

    // Free memory
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);

    return 0;
}
