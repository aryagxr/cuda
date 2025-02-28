#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>

#include "hgemv.cu"


int main(){

    half* h_A = (half*)malloc(M * N * sizeof(half));
    half* h_v = (half*)malloc(N * sizeof(half));
    half* h_C = (half*)malloc(M * sizeof(half));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float random_value = 1.0f + static_cast<float>(rand()) / (RAND_MAX / 9.0f);  // Random between 1 and 10
            h_A[i * N + j] = __float2half(random_value);  // Convert to FP16 and store in A
        }
    }
    
    for (int i = 0; i < N; i++) {
        float random_value = 1.0f + static_cast<float>(rand()) / (RAND_MAX / 9.0f);  // Random between 1 and 10
        h_v[i] = __float2half(random_value);
    }

    half* d_A;
    half* d_v;
    half* d_C;
    cudaMalloc(&d_A, M * N * sizeof(half));
    cudaMalloc(&d_v, N * sizeof(half));
    cudaMalloc(&d_C, M * sizeof(half));

    cudaMemcpy(d_A, h_A, M * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, N * sizeof(half), cudaMemcpyHostToDevice);

    run_kernel1_hgemv_fp16(d_A, d_v, d_C);

    cudaMemcpy(h_C, d_C, M * sizeof(half), cudaMemcpyDeviceToHost);

    // Print first few results (just to verify the kernel worked)
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << __half2float(h_C[i]) << std::endl;
    }

    // Clean up
    free(h_A);
    free(h_v);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_C);

    return 0;

}