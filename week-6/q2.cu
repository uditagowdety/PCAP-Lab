#include <stdio.h>

__global__ void selection_sort(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = idx; i < n - 1; i++) {
            int min_idx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[min_idx]) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                int temp = arr[i];
                arr[i] = arr[min_idx];
                arr[min_idx] = temp;
            }
        }
    }
}

void launch_selection_sort(int *arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    selection_sort<<<num_blocks, block_size>>>(d_arr, n);
    
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    launch_selection_sort(arr, n);

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}