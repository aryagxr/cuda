// matrix multiplication using cuBLAS

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <curand.h>

#define N 1024


int main(){
    
    size_t matrix_size = N*N*sizeof(float);
    float alpha = 1.0f;
    float beta = 0.0f;

    // allocate host memory
    float *h_A = (float*)malloc(matrix_size);
    float *h_B = (float*)malloc(matrix_size);
    float *h_C = (float*)malloc(matrix_size);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    
    for (int i = 0; i<N*N; i++){
        h_A[i] = (i % N) + 1;   // 1, 2, 3, ..., N, columnwise
        h_B[i] = (i / N) + 1;  // 1,2,3...N rowwise
    }

    

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMatrix(N, N, sizeof(float), h_A, N, d_A, N);
    cublasSetMatrix(N, N, sizeof(float), h_B, N, d_B, N);
    // cublasSetMatrix(N, N, sizeof(float), h_C, N, d_C, N);
    cudaMemset(d_C, 0, matrix_size);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Ensure kernel execution is finished

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";


    
    cublasGetMatrix(N, N, sizeof(float), d_C, N, h_C, N);
    cublasDestroy(handle);

    printf("Successful");
    printf("Sample result C[0][0] = %f\n", h_C[0]);

    // free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;

}