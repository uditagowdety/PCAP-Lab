#include <stdio.h>

__global__ void conv1d(float *input, float *kernel, float *output, int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_kernel = kernel_size / 2;
    if (idx >= half_kernel && idx < input_size - half_kernel) {
        float sum = 0.0f;
        for (int k = -half_kernel; k <= half_kernel; k++) {
            sum += input[idx + k] * kernel[half_kernel + k];
        }
        output[idx] = sum;
    }
}

void launch_conv1d(float *input, float *kernel, float *output, int input_size, int kernel_size) {
    float *d_input, *d_kernel, *d_output;
    int size_input = input_size * sizeof(float);
    int size_kernel = kernel_size * sizeof(float);
    int size_output = input_size * sizeof(float);
    cudaMalloc((void**)&d_input, size_input);
    cudaMalloc((void**)&d_kernel, size_kernel);
    cudaMalloc((void**)&d_output, size_output);
    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, size_kernel, cudaMemcpyHostToDevice);
    int block_size = 256;
    int num_blocks = (input_size + block_size - 1) / block_size;
    conv1d<<<num_blocks, block_size>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    int input_size = 10;
    int kernel_size = 3;
    float input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float kernel[3] = {0.2, 0.5, 0.2};
    float output[10] = {0};

    launch_conv1d(input, kernel, output, input_size, kernel_size);

    for (int i = 0; i < input_size; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");
    return 0;
}
