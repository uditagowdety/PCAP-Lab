#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_SIZE 3

__constant__ int d_filter[FILTER_SIZE];

__global__ void convolution_1d(int* d_input, int* d_output, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= FILTER_SIZE - 1 && idx < input_size - FILTER_SIZE + 1) {
        int result = 0;
        for (int i = 0; i < FILTER_SIZE; i++) {
            result += d_input[idx + i - FILTER_SIZE / 2] * d_filter[i];
        }
        d_output[idx] = result;
    }
}

int main() {
    int input_size;
    printf("Enter input size: ");
    scanf("%d", &input_size);

    int* input = (int*)malloc(input_size * sizeof(int));
    int* output = (int*)malloc(input_size * sizeof(int));
    int filter[FILTER_SIZE] = {1, 2, 1}; // Example filter

    printf("Enter input array:\n");
    for (int i = 0; i < input_size; i++) {
        scanf("%d", &input[i]);
    }

    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_output, input_size * sizeof(int));

    cudaMemcpyToSymbol(d_filter, filter, FILTER_SIZE * sizeof(int));
    cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (input_size + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_size);

    cudaMemcpy(output, d_output, input_size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output array:\n");
    for (int i = 0; i < input_size; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
