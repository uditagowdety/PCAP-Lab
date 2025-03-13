#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

__global__ void count_word(char* sentence, char* word, int* count, int s_len, int w_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= s_len - w_len) {
        bool match = true;
        for (int j = 0; j < w_len && match; j++)
            if (sentence[i + j] != word[j]) match = false;
        if (match) atomicAdd(count, 1);
    }
}

int main() {
    char sentence[] = "this is a test sentence for testing", word[] = "test";
    int count = 0, s_len = strlen(sentence), w_len = strlen(word);
    char *d_sentence, *d_word;
    int *d_count;

    cudaMalloc(&d_sentence, s_len + 1);
    cudaMalloc(&d_word, w_len + 1);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_sentence, sentence, s_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, w_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    count_word<<<(s_len / 256) + 1, 256>>>(d_sentence, d_word, d_count, s_len, w_len);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("word '%s' appears %d times\n", word, count);

    cudaFree(d_sentence); cudaFree(d_word); cudaFree(d_count);
    return 0;
}
