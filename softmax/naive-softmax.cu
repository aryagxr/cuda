#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

// Naive softmax of M*N matrix
// Parallelizing over rows of matrix
// One thread for each row
// Assuming M 1024 rows, --> 1024 threads
// If one block has x threads
// blocks needed --> ceil(M / x)


#define M 1024
#define N 32768


__global__ void naive_softmax(float *X, float *P, int m, int n){


    // thread index for each row
    int row = threadIdx.x + (blockDim.x * blockIdx.x);

    if (row < M){

        float x_max = -INFINITY;
        float norm = 0.0f;

        // find max value in current row
        for(int col = 0; col < n; col++){
            int i = row * n + col;
            x_max = max(X[i], x_max);
        }

        // compute norm factor (denominator of softmax)
        for(int col = 0; col < n; col++){
            int i = row * n + col;
            norm += expf(X[i]- x_max);
        }

        // compute softmax probability
        for(int col = 0; col < n; col++){
            int i = row * n + col;
            P[i] = (expf(X[i] - x_max)) / norm;
        }
    }
}


int main(){

    size_t matrix_size = M*N*sizeof(float);

    float *X_input, *P_output;
    float *D_input, *D_output;

    X_input = (float*)malloc(matrix_size);
    P_output = (float*)malloc(matrix_size);

    cudaMalloc((void**)&D_input, matrix_size);
    cudaMalloc((void**)&D_output, matrix_size);

    for(int a = 0; a<M*N; a++){
        X_input[a] = ((float)rand() / RAND_MAX) * 10.0f;
    }

    cudaMemcpy(D_input, X_input, matrix_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid(ceil(M/threadsPerBlock.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);
    

    naive_softmax<<<blocksPerGrid, threadsPerBlock>>>(D_input, D_output, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(P_output, D_output, matrix_size, cudaMemcpyDeviceToHost);

    printf("Softmax result (first row only):\n");
    for (int j = 0; j < 10; j++) {  // Print only first 10 values for brevity
        printf("%f ", P_output[j]);
    }
    printf("...\n");

    cudaFree(D_input); cudaFree(D_output);
    free(X_input); free(P_output);

    return 0;
    
}


