#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

__global__ void generate_string(char* s, char* rs, int s_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < s_len) {
        for(int i = 0; i <= index; i++) rs[index * s_len + i] = s[i];
    }
}

int main() {
    char s[] = "PCAP";
    int s_len = strlen(s), rs_len = s_len * s_len;
    char *d_s, *d_rs, rs[rs_len];

    cudaMalloc(&d_s, s_len * sizeof(char));
    cudaMalloc(&d_rs, rs_len * sizeof(char));
    cudaMemcpy(d_s, s, s_len * sizeof(char), cudaMemcpyHostToDevice);

    generate_string<<<(s_len / 256) + 1, 256>>>(d_s, d_rs, s_len);
    cudaMemcpy(rs, d_rs, rs_len * sizeof(char), cudaMemcpyDeviceToHost);

    for(int i = 0; i < rs_len; i++) if(rs[i]) printf("%c", rs[i]);
    printf("\n");

    cudaFree(d_s); cudaFree(d_rs);
    return 0;
}
