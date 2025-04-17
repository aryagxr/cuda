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

__global__ void kernel1_swish_fp32(float* in, float* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (tidx < n){
        float x = in[tidx];
        out[tidx] = x / (1.0f + expf(-x));
    }
}

__global__ void kernel2_swish_4fp32(float* in, float* out, int n){
    int tidx = (threadIdx.x + (blockDim.x * blockIdx.x)) * 4; // 0, 4, 8, 12...
    if(tidx < n){
        float4 x = FLOAT4(in[tidx]);
        float4 y;
        y.x = x.x / (1.0f + expf(-x.x));
        y.y = x.y / (1.0f + expf(-x.y));
        y.z = x.z / (1.0f + expf(-x.z));
        y.w = x.w / (1.0f + expf(-x.w));

        FLOAT4(out[tidx]) = y;
    }
}


__global__ void kernel3_swish_fp16(half* in, half* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (tidx < n){
        half x = in[tidx];
        float x_float = __half2float(x);
        float result = x_float / (1.0f + expf(-x_float));
        out[tidx] = __float2half(result);
    }
}


int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *h_output2 = (float*)malloc(bytes);
    
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_input[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
    
    float *d_input, *d_output, *d_output2;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_output2, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_vec = (N + threadsPerBlock*4 - 1) / (threadsPerBlock*4);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel1_swish_fp32<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    std::cout << "Swish kernel1 execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (N * sizeof(float)) / (milliseconds * 1.0e6) << " GB/s" << std::endl;
    
    cudaEventRecord(start);
    kernel2_swish_4fp32<<<blocksPerGrid_vec, threadsPerBlock>>>(d_input, d_output2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);
    cudaMemcpy(h_output2, d_output2, bytes, cudaMemcpyDeviceToHost);
    
    std::cout << "Swish kernel2 vectorized execution time: " << milliseconds2 << " ms" << std::endl;
    std::cout << "Throughput: " << (N * sizeof(float)) / (milliseconds2 * 1.0e6) << " GB/s" << std::endl;
    
    std::cout << "\nSample results kernel1:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Input: " << h_input[i] << " → Swish: " << h_output[i] << std::endl;
    }
    
    std::cout << "\nSample results kernel2:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Input: " << h_input[i] << " → Swish: " << h_output2[i] << std::endl;
    }
    
    
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output2);
    free(h_input);
    free(h_output);
    free(h_output2);
    
    return 0;
}