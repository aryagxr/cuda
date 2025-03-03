#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;


#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16



/*  Kernel 1: 16*16*16 matmul in a single warp */

__global__ void kernel1_wmma_fp16(half* A, half* B, float* C){

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);

}




int main() {
    // Allocate and initialize host matrices
    half* h_A = new half[MATRIX_M * MATRIX_K];
    half* h_B = new half[MATRIX_K * MATRIX_N];
    float* h_C = new float[MATRIX_M * MATRIX_N];
    
    for (int i = 0; i < MATRIX_M * MATRIX_K; ++i) h_A[i] = __float2half(i + 1);
    for (int i = 0; i < MATRIX_K * MATRIX_N; ++i) h_B[i] = __float2half(i + 1);
    for (int i = 0; i < MATRIX_M * MATRIX_N; ++i) h_C[i] = 0.0f;

    /*
    std::cout << "Input Matrix A:\n";
    for (int i = 0; i < MATRIX_M; ++i) {
        for (int j = 0; j < MATRIX_K; ++j) {
            std::cout << __half2float(h_A[i * MATRIX_K + j]) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nInput Matrix B:\n";
    for (int i = 0; i < MATRIX_K; ++i) {
        for (int j = 0; j < MATRIX_N; ++j) {
            std::cout << __half2float(h_B[i * MATRIX_N + j]) << " ";
        }
        std::cout << "\n";
    }
    */
    
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, MATRIX_M * MATRIX_K * sizeof(half));
    cudaMalloc(&d_B, MATRIX_K * MATRIX_N * sizeof(half));
    cudaMalloc(&d_C, MATRIX_M * MATRIX_N * sizeof(float));

    
    cudaMemcpy(d_A, h_A, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice);

    
    kernel1_wmma_fp16<<<1, 32>>>(d_A, d_B, d_C);

    
    cudaMemcpy(h_C, d_C, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    std::cout << "Output Matrix C:\n";
    for (int i = 0; i < MATRIX_M; ++i) {
        for (int j = 0; j < MATRIX_N; ++j) {
            std::cout << h_C[i * MATRIX_N + j] << " ";
        }
        std::cout << "\n";
    }
    */

    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}


