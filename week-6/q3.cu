#include <stdio.h>

__global__ void even_odd_sort(int *arr, int n, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        int even_idx = idx * 2;
        if (phase == 0 && even_idx + 1 < n) {
            if (arr[even_idx] > arr[even_idx + 1]) {
                int temp = arr[even_idx];
                arr[even_idx] = arr[even_idx + 1];
                arr[even_idx + 1] = temp;
            }
        } else if (phase == 1 && even_idx + 1 < n) {
            if (arr[even_idx + 1] > arr[even_idx + 2] && even_idx + 2 < n) {
                int temp = arr[even_idx + 1];
                arr[even_idx + 1] = arr[even_idx + 2];
                arr[even_idx + 2] = temp;
            }
        }
    }
}

void launch_even_odd_sort(int *arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    for (int phase = 0; phase < 2; phase++) {
        even_odd_sort<<<num_blocks, block_size>>>(d_arr, n, phase);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    launch_even_odd_sort(arr, n);

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}