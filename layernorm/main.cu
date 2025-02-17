#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#include "kernels/vectorized-layernorm.cu"

int main(){

    int M = 1024;
    int N = 1024;

    size_t matrix_size = M*N*sizeof(float);
    float *X_input, *P_output;
    float *D_input, *D_output;

    X_input = (float*)malloc(matrix_size);
    P_output = (float*)malloc(matrix_size);

    for(int i = 0; i < M*N; i++){
        X_input[i] = i+1;
    }

    cudaMalloc((void**)&D_input, matrix_size);
    cudaMalloc((void**)&D_output, matrix_size);

    cudaMemcpy(D_input, X_input, matrix_size, cudaMemcpyHostToDevice);

    run_vect_ln(D_input, D_output, M, N);

    cudaMemcpy(P_output, D_output, matrix_size, cudaMemcpyDeviceToHost);

    printf("Input matrix: \n");
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            printf("%f", X_input[i*N+j]);
        }
        printf("\n");
    }

    printf("Output matrix: \n");
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            printf("%f", P_output[i*N+j]);
        }
        printf("\n");
    }

    free(P_output); free(X_input);
    cudaFree(D_input); cudaFree(D_output);

}