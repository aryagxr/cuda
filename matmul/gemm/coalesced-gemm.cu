#include <iostream>
#include <cuda_runtime.h>

#include <iostream>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))


__global__ void gemm_coalesced(float *A, float *B, float *C, float alpha, float beta, int M, int N, int K){
    const int row = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
    const int col = blockIdx.y * blockDim.y + (threadIdx.y / blockDim.y);

    if(row < M && col < N){
        float acc = 0.0;
        for(int i = 0; i < K; ++i){
            acc += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * acc + C[row * N + col] * beta;
    }
}

#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;

    size_t A_bytes = M * K * sizeof(float);
    size_t B_bytes = K * N * sizeof(float);
    size_t C_bytes = M * N * sizeof(float);

    float *h_A = (float*)malloc(A_bytes);
    float *h_B = (float*)malloc(B_bytes);
    float *h_C = (float*)malloc(C_bytes);
    
    srand(42);
    for (int i = 0; i < M * K; ++i)
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < M * N; ++i)
        h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_bytes);
    cudaMalloc(&d_B, B_bytes);
    cudaMalloc(&d_C, C_bytes);

    cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, C_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);
    gemm_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost);

    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    
    // Print a small subset of results
    std::cout << "Result matrix (top-left corner):" << std::endl;
    for (int i = 0; i < 4 && i < M; ++i) {
        for (int j = 0; j < 4 && j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}

