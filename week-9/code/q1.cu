#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void spmv(int* row_ptr, int* col_ind, int* values, int* x, int* y, int rows){
    int row=blockDim.x*blockIdx.x+threadIdx.x;

    if(row<rows){
        int sum=0.0;

        for (int j=row_ptr[row];j<row_ptr[row+1];j++){
            sum+=values[j]*x[col_ind[j]];
        }

        y[row]=sum;
    }
}

void toCSR(int* matrix, int m, int n, int* values, int* col_indices, int* row_ptr, int* nnz) {
    int count = 0;
    row_ptr[0] = 0;  // First row starts at index 0
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i * n + j] != 0) {
                values[count] = matrix[i * n + j];
                col_indices[count] = j;
                count++;
            }
        }
        row_ptr[i + 1] = count;  // Store where the next row starts
    }
    *nnz = count;
}

int main(){
    int m,n;
    printf("enter matrix dimensions (row, col): ");
    scanf("%d%d",&m,&n);

    int* matrix=(int*)malloc(m*n*sizeof(int));
    printf("enter matrix elements:\n");
    for(int i=0;i<m*n;i++){
        scanf("%d",&matrix[i]);
    }

    int* row_ptr=(int*)malloc((m+1)*sizeof(int));
    int* col_ind=(int*)malloc(m*n*sizeof(int));
    int* values=(int*)malloc(m*n*sizeof(int));
    int nnz;

    toCSR(matrix, m, n, values, col_ind, row_ptr, &nnz);

    printf("csr format representation:\n");
    printf("values: { ");
    for(int i=0;i<nnz;i++) printf("%d ",values[i]);

    printf("}\ncolumn indices: { ");
    for(int i=0;i<nnz;i++) printf("%d ",col_ind[i]);

    printf("}\nrow pointers: { ");
    for(int i=0;i<=m;i++) printf("%d ",row_ptr[i]);
    printf("}\n");

    int* x = (int*)malloc(n * sizeof(int));
    printf("\nenter vector elements (size %d):\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &x[i]);
    }

    int *d_row_ptr, *d_col_indices;
    int *d_values, *d_x, *d_y;
    int* y = (int*)malloc(m * sizeof(int));

    cudaMalloc((void**)&d_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_indices, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(int));
    cudaMalloc((void**)&d_x, n * sizeof(int));
    cudaMalloc((void**)&d_y, m * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    spmv<<<numBlocks, threadsPerBlock>>>(d_row_ptr, d_col_indices, d_values, d_x, d_y, m);

    cudaMemcpy(y, d_y, m * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nresulting vector Y:\n");
    for (int i = 0; i < m; i++) {
        printf("%d\n ", y[i]);
    }
    printf("\n");

    free(matrix);
    free(row_ptr);
    free(col_ind);
    free(values);
    free(x);
    free(y);
    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}