#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <iostream>

#define sqrt2overPI 0.7978845608
#define k 0.044715

__global__ void kernel1_gelu_fp32(float* in, float* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);

    if(tidx < n){
        float x = in[tidx];
        out[tidx] = 0.5f * x * (1.0f + tanh(sqrt2overPI * (x + k * (x * x * x))));
    }
}


int main(){
    const int N = 1024;
    size_t fp32_size = N * sizeof(float);

    float *X, *P;
    float *dx, *dp;
    
    X = (float*)malloc(fp32_size);
    P = (float*)malloc(fp32_size);
    

    cudaMalloc((void**)&dx, fp32_size);
    cudaMalloc((void**)&dp, fp32_size);
    
    
    for (int i = 0; i < N; ++i) {
        X[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }

    std::cout << "Input matrix:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << X[i] << " ";
    }
    std::cout << std::endl;

    cudaMemcpy(dx, X, fp32_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock-1) / threadsPerBlock;
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);
    

    kernel1_gelu_fp32<<<blocksPerGrid, threadsPerBlock>>>(dx, dp, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Scalar FP32 Kernel1 execution time: " << ms << " ms\n";

    cudaMemcpy(P, dp, fp32_size, cudaMemcpyDeviceToHost);


    


    for (int i = 0; i < 10; ++i) {
        std::cout << P[i] << " ";
    }
    std::cout << std::endl;

    


    cudaFree(dx); cudaFree(dp);
    free(X); free(P); 

    return 0;
    
}

