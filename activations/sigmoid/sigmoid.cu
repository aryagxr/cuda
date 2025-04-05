#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void kernel1_sigmoid_fp32(float* in, float* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < n){
        float x = in[tidx];
        out[tidx] = 1 / (1 + expf(-x));
    }
}

__global__ void kernel2_sigmoid_4fp32(float* in, float* out, int n){
    int tidx = (threadIdx.x + (blockDim.x * blockIdx.x)) * 4;
    if(tidx < n){
        float4 x = FLOAT4(in[tidx]);
        float4 y;
        y.x = 1 / (1 + expf(-x.x));
        y.y = 1 / (1 + expf(-x.y));
        y.z = 1 / (1 + expf(-x.z));
        y.w = 1 / (1 + expf(-x.w));

        FLOAT4(out[tidx]) = y;
    }
}




int main(){
    const int N = 1024;
    size_t fp32_size = N * sizeof(float);

    float *X, *P, *P2;
    float *dx, *dp, *dp2;
    
    X = (float*)malloc(fp32_size);
    P = (float*)malloc(fp32_size);
    P2 = (float*)malloc(fp32_size);

    cudaMalloc((void**)&dx, fp32_size);
    cudaMalloc((void**)&dp, fp32_size);
    cudaMalloc((void**)&dp2, fp32_size);
    
    for (int i = 0; i < N; ++i) {
        X[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }


    cudaMemcpy(dx, X, fp32_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock-1) / threadsPerBlock;
    int blocksPerGrid_vec = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);
    

    kernel1_sigmoid_fp32<<<blocksPerGrid, threadsPerBlock>>>(dx, dp, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Scalar FP32 Kernel1 execution time: " << ms << " ms\n";

    cudaMemcpy(P, dp, fp32_size, cudaMemcpyDeviceToHost);


    // Run vectorized FP32 * 4 kernel
    cudaEventRecord(start);
    kernel2_sigmoid_4fp32<<<blocksPerGrid_vec, threadsPerBlock>>>(dx, dp2, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Vectorized FP32*4 Kernel2 execution time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(P2, dp2, fp32_size, cudaMemcpyDeviceToHost);


    for (int i = 0; i < 10; ++i) {
        std::cout << P[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << P2[i] << " ";
    }


    cudaFree(dx); cudaFree(dp); cudaFree(dp2);
    free(X); free(P); free(P2);

    return 0;
    
}



