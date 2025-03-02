#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>

#include "hgemv.cu"


/*  TODO:
    Organize code to switch between datatypes
*/


// typedef half hp;
typedef __nv_bfloat16 hp;


int main(){

    hp* h_A = (hp*)malloc(M * N * sizeof(hp));
    hp* h_v = (hp*)malloc(N * sizeof(hp));
    hp* h_C = (hp*)malloc(M * sizeof(hp));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float random_value = 1.0f + static_cast<float>(rand()) / (RAND_MAX / 9.0f);  // Random between 1 and 10
            h_A[i * N + j] = __float2bfloat16(random_value);  // Convert to FP16 and store in A
        }
    }
    
    for (int i = 0; i < N; i++) {
        float random_value = 1.0f + static_cast<float>(rand()) / (RAND_MAX / 9.0f);  // Random between 1 and 10
        h_v[i] = __float2bfloat16(random_value);
    }

    hp* d_A;
    hp* d_v;
    hp* d_C;
    cudaMalloc(&d_A, M * N * sizeof(hp));
    cudaMalloc(&d_v, N * sizeof(hp));
    cudaMalloc(&d_C, M * sizeof(hp));

    cudaMemcpy(d_A, h_A, M * N * sizeof(hp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, N * sizeof(hp), cudaMemcpyHostToDevice);

    run_kernel2_hgemv_bf16(d_A, d_v, d_C);

    cudaMemcpy(h_C, d_C, M * sizeof(hp), cudaMemcpyDeviceToHost);

    // Print first few results (just to verify the kernel worked)
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << __bfloat162float(h_C[i]) << std::endl;
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