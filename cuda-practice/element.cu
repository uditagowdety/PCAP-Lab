#include<stdio.h>
#include "cuda_runtime.h"

__global__ void element_add(int *a, int*b, int* c){
    int index=threadIdx.x;
    c[index]=a[index]+b[index];
}

int main(){
    int arr1[]={1,2,3,4,5};
    int arr2[]={6,7,8,9,10};
    int result[5];
    int size=sizeof(int)*5;

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,arr1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,arr2,size,cudaMemcpyHostToDevice);

    element_add<<<1,5>>>(d_a,d_b,d_c);

    cudaMemcpy(result,d_c,size,cudaMemcpyDeviceToHost);

    printf("result array:  ");
    for (int i=0;i<5;i++) printf("%d ",result[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}