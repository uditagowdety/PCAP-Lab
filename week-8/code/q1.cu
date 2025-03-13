#include<stdio.h>
#include<cuda_runtime.h>

__global__ void rowAdd(int* a, int* b, int* c, int m, int n){
    // int row=threadIdx.x+blockIdx.x*blockDim.x;
    // int row=blockIdx.y*blockDim.y+threadIdx.y;
    int row=blockIdx.x;
    // int col=blockIdx.x*blockDim.x+threadIdx.x;

    if(row<m){
        for(int col=0;col<m;col++) c[row*n+col]=a[row*n+col]+b[row*n+col];
    }
}

__global__ void colAdd(int* a, int* b, int* c, int m, int n){
    // int col=threadIdx.x+blockIdx.x*blockDim.x;
    int col=blockIdx.x;
    if(col<n){
        for(int row=0;row<m;row++){
            c[row*n+col]=a[row*n+col]+b[row*n+col];
        }
    }
}

__global__ void elemAdd(int* a, int* b, int* c, int m, int n){
    int idx=threadIdx.x+blockIdx.x+blockDim.x;
    if(idx<m*n){
        int row=idx/n;
        int col=idx%n;
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

int main(){
    int m,n;
    printf("enter dimensions for m and n: ");
    scanf("%d%d",&m, &n);

    int size=m*n*sizeof(int);
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    a=(int*)malloc(size);
    b=(int*)malloc(size);
    c=(int*)malloc(size);

    printf("enter matrix A:\n");
    for(int i=0;i<m*n;i++){
        scanf("%d",&a[i]);
    }

    printf("enter matrix B:\n");
    for(int i=0;i<m*n;i++){
        scanf("%d",&b[i]);
    }

    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1);
    dim3 numBlocks(m);

    rowAdd<<<numBlocks, threadsPerBlock>>>(d_a,d_b,d_c,m,n);

    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

    printf("resultant row added matrix:\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%d\t",c[i*n+j]);
        }
        printf("\n");
    }

    // dim3 threadsPerBlock(1);
    // dim3 numBlocks(n);

    colAdd<<<n,1>>>(d_a,d_b,d_c,m,n);

    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

    printf("resultant column added matrix:\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%d\t",c[i*n+j]);
        }
        printf("\n");
    }



    elemAdd<<<(m+255)/256,256>>>(d_a,d_b,d_c,m,n);

    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

    printf("resultant element added matrix:\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%d\t",c[i*n+j]);
        }
        printf("\n");
    }



    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}