#include <stdio.h>
#include <cuda_runtime.h>

__global__ void rowMul(int* a, int* b, int* c, int m, int n, int p){
    int row = blockIdx.x;
    if(row < m){
        for(int col = 0; col < p; col++){
            int sum = 0;
            for(int k = 0; k < n; k++){
                sum += a[row * n + k] * b[k * p + col];
            }
            c[row * p + col] = sum;
        }
    }
}

__global__ void colMul(int* a, int* b, int* c, int m, int n, int p){
    int col = blockIdx.x;
    if(col < p){
        for(int row = 0; row < m; row++){
            int sum = 0;
            for(int k = 0; k < n; k++){
                sum += a[row * n + k] * b[k * p + col];
            }
            c[row * p + col] = sum;
        }
    }
}

__global__ void elemMul(int* a, int* b, int* c, int m, int n, int p){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < m * p){
        int row = idx / p;
        int col = idx % p;
        int sum = 0;
        for(int k = 0; k < n; k++){
            sum += a[row * n + k] * b[k * p + col];
        }
        c[row * p + col] = sum;
    }
}

int main() {
    int m, n, q, p;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    printf("Enter matrix 1 dimensions (m x n): ");
    scanf("%d%d", &m, &n);

    printf("Enter matrix 2 dimensions (q x p): ");
    scanf("%d%d", &q, &p);

    if (n != q) {
        printf("Matrix dimensions are not compatible for multiplication.\n");
        return -1;
    }

    a = (int*)malloc(m * n * sizeof(int));
    b = (int*)malloc(q * p * sizeof(int));
    c = (int*)malloc(m * p * sizeof(int));

    printf("Enter matrix 1 values:\n");
    for(int i = 0; i < m * n; i++) {
        scanf("%d", &a[i]);
    }

    printf("Enter matrix 2 values:\n");
    for(int i = 0; i < q * p; i++) {
        scanf("%d", &b[i]);
    }

    cudaMalloc((void**)&d_a, m * n * sizeof(int));
    cudaMalloc((void**)&d_b, q * p * sizeof(int));
    cudaMalloc((void**)&d_c, m * p * sizeof(int));

    cudaMemcpy(d_a, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, q * p * sizeof(int), cudaMemcpyHostToDevice);

    rowMul<<<m, 1>>>(d_a, d_b, d_c, m, n, p);
    cudaMemcpy(c, d_c, m * p * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Resultant matrix C (Row-wise):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%d ", c[i * p + j]);
        }
        printf("\n");
    }

    colMul<<<p, 1>>>(d_a, d_b, d_c, m, n, p);
    cudaMemcpy(c, d_c, m * p * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Resultant matrix C (Column-wise):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%d ", c[i * p + j]);
        }
        printf("\n");
    }

    int blockSize = 256;  
    dim3 threadsPerBlock(blockSize);
    dim3 numBlocks((m * p + blockSize - 1) / blockSize);
    elemMul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);
    cudaMemcpy(c, d_c, m * p * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Resultant matrix C (Element-wise):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%d ", c[i * p + j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
