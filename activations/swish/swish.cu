#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

__global__ void kernel1_swish_fp32(float* in, float* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (tidx < n){
        float x = in[tidx];
        out[tidx] = x / (1.0f + expf(-x));
    }
}


int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_input[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
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
    
    std::cout << "Swish kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (N * sizeof(float)) / (milliseconds * 1.0e6) << " GB/s" << std::endl;
    
    std::cout << "\nSample results:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Input: " << h_input[i] << " → Swish: " << h_output[i] << std::endl;
    }
    
    std::cout << "\nVerification (CPU vs GPU):" << std::endl;
    bool verification_passed = true;
    for (int i = 0; i < 5; ++i) {
        float cpu_result = h_input[i] / (1.0f + expf(-h_input[i]));
        float error = fabs(cpu_result - h_output[i]);
        std::cout << "CPU: " << cpu_result << " GPU: " << h_output[i];
        std::cout << " (error: " << error << ")" << std::endl;
        if (error > 1e-5) {
            verification_passed = false;
        }
    }
    std::cout << "\nVerification " << (verification_passed ? "PASSED" : "FAILED") << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}