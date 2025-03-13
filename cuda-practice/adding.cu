#include<stdio.h>
#include "cuda_runtime.h"

__global__ void add(int* a, int* b, int* c){
    *c=*a+*b;
    printf("inside device: %d + %d = %d\n",*a, *b, *c);
}

int main(){
    int a,b,c;
    int size=sizeof(int);
    a=3;
    b=5;

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,&a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,size,cudaMemcpyHostToDevice);

    add<<<1,1>>>(d_a,d_b,d_c);

    cudaMemcpy(&c,d_c,size,cudaMemcpyDeviceToHost);

    printf("result from device: %d\n",c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}