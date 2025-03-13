#include<cuda_runtime.h>
#include<stdio.h>
#include<device_launch_parameters.h>
#include<stdlib.h>

__global__ void transpose(int* a, int* t, int m, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // Row index
    int j = threadIdx.y + blockIdx.y * blockDim.y;  // Column index
    
    if (i < m && j < n) {  // Ensure within bounds
        t[j * m + i] = a[i * n + j];  // Transpose logic
    }
}


int main(){
    int *a, *t, m,n;
    int *d_a, *d_t;

    printf("enter the value of m: ");
    scanf("%d", &m);

    printf("enter the value of n: ");
    scanf("%d", &n);

    int size=sizeof(int)*m*n;

    a=(int*)malloc(size);
    t=(int*)malloc(size);

    printf("enter input matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &a[i * n + j]);
        }
    }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_t, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + 15) / 16, (n + 15) / 16);  // Grid size to handle the matrix
   
    transpose<<<numBlocks,threadsPerBlock>>>(d_a,d_t,m,n);

    cudaMemcpy(t,d_t,size,cudaMemcpyDeviceToHost);

    printf("result vector is:\n");
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            printf("%d\t",t[i*m+j]);
        }
        printf("\n");
    }

    getchar();
    cudaFree(d_a);
    cudaFree(d_t);

    return 0;
}