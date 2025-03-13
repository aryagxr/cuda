#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>


// The [0] accesses the first float4 that starts at the memory address &x[idx]
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])


/*  Kernel 1: FP32 */
__global__ void kernel1_relu_fp32(float* in, float* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (tidx < n){
        out[tidx] = fmaxf(0.0, in[tidx]);
    }
    __syncthreads();

}


/*  Kernel 2: Vectorized FP32 * 4 */
__global__ void kernel2_relu_4fp32(float* in, float* out, int n){

    int tidx = (threadIdx.x + (blockDim.x * blockIdx.x)) * 4; // 0, 4, 8, 12...
    if(tidx < n){
        float4 x = FLOAT4(in[tidx]);
        float4 y;
        y.x = fmaxf(0.0, x.x);
        y.y = fmaxf(0.0, x.y);
        y.z = fmaxf(0.0, x.z);
        y.w = fmaxf(0.0, x.w);

        FLOAT4(out[tidx]) = y;
    }
}


/* Kernel 3: FP16 Half Precision */
__global__ void kernel3_relu_fp16(half* in, half* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    half zero = __float2half(0.0f);
    if(tidx < n){
        out[tidx] = fmaxf(zero, in[tidx]);
    }
    __syncthreads();
}


/* Kernel 4: Vectorized FP16 * 2 */
__global__ void kernel4_relu_2fp16(half* in, half* out, int n){
    int tidx = (threadIdx.x + (blockDim.x * blockIdx.x) * 2); // 0, 2, 4, 6...
    half zero = __float2half(0.0f);
    if(tidx < n){
        half2 x = HALF2(in[tidx]);
        half2 y;
        y.x = __hmax(zero, x.x);
        y.y = __hmax(zero, x.y);
        HALF2(out[tidx]) = y;
    }
}




/* Host helper Functions */

void initialize_data(float* in_fp32, half* in_fp16, int size) {
    unsigned long long seed = 1234;
    
    for (int i = 0; i < size; ++i) {
        in_fp32[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f; // Random float in range [-1, 1]
    }
    
    for (int i = 0; i < size; ++i) {
        in_fp16[i] = __float2half((rand() / float(RAND_MAX)) * 2.0f - 1.0f); // Random float in range [-1, 1]
    }
}

void print_float_array(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void print_half_array(half* arr, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << __half2float(arr[i]) << " ";
    }
    std::cout << std::endl;
}


int main(){
    const int N = 1024;

    // Host arrays
    float* h_in_fp32 = new float[N];
    float* h_out_fp32_1 = new float[N];  
    float* h_out_fp32_2 = new float[N];  
    half* h_in_fp16 = new half[N];
    half* h_out_fp16_1 = new half[N];    
    half* h_out_fp16_2 = new half[N];    

    initialize_data(h_in_fp32, h_in_fp16, N);

    float *d_in_fp32, *d_out_fp32_1, *d_out_fp32_2;
    half *d_in_fp16, *d_out_fp16_1, *d_out_fp16_2;
    cudaMalloc((void**)&d_in_fp32, N * sizeof(float));
    cudaMalloc((void**)&d_out_fp32_1, N * sizeof(float));
    cudaMalloc((void**)&d_out_fp32_2, N * sizeof(float));
    cudaMalloc((void**)&d_in_fp16, N * sizeof(half));
    cudaMalloc((void**)&d_out_fp16_1, N * sizeof(half));
    cudaMalloc((void**)&d_out_fp16_2, N * sizeof(half));

    cudaMemcpy(d_in_fp32, h_in_fp32, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_fp16, h_in_fp16, N * sizeof(half), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    kernel1_relu_fp32<<<gridSize, blockSize>>>(d_in_fp32, d_out_fp32_1, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time1 = 0;
    cudaEventElapsedTime(&time1, start, stop);

    cudaEventRecord(start);
    kernel2_relu_4fp32<<<gridSize, blockSize>>>(d_in_fp32, d_out_fp32_2, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time2 = 0;
    cudaEventElapsedTime(&time2, start, stop);

    cudaEventRecord(start);
    kernel3_relu_fp16<<<gridSize, blockSize>>>(d_in_fp16, d_out_fp16_1, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time3 = 0;
    cudaEventElapsedTime(&time3, start, stop);

    cudaEventRecord(start);
    kernel4_relu_2fp16<<<gridSize, blockSize>>>(d_in_fp16, d_out_fp16_2, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time4 = 0;
    cudaEventElapsedTime(&time4, start, stop);

    cudaMemcpy(h_out_fp32_1, d_out_fp32_1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fp32_2, d_out_fp32_2, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fp16_1, d_out_fp16_1, N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fp16_2, d_out_fp16_2, N * sizeof(half), cudaMemcpyDeviceToHost);

    // Print the input matrices before kernel launch
    std::cout << "Input FP32 matrix (first 5 elements): ";
    print_float_array(h_in_fp32, 5);

    std::cout << "Input FP16 matrix (first 5 elements): ";
    print_half_array(h_in_fp16, 5);

    // Print the output matrices after kernel execution
    std::cout << "Output of kernel 1 (FP32, first 5 elements): ";
    print_float_array(h_out_fp32_1, 5);

    std::cout << "Output of kernel 2 (FP32 * 4, first 5 elements): ";
    print_float_array(h_out_fp32_2, 5);

    std::cout << "Output of kernel 3 (FP16, first 5 elements): ";
    print_half_array(h_out_fp16_1, 5);

    std::cout << "Output of kernel 4 (FP16 * 2, first 5 elements): ";
    print_half_array(h_out_fp16_2, 5);

    std::cout << "Kernel 1 execution time: " << time1 << " ms" << std::endl;
    std::cout << "Kernel 2 execution time: " << time2 << " ms" << std::endl;
    std::cout << "Kernel 3 execution time: " << time3 << " ms" << std::endl;
    std::cout << "Kernel 4 execution time: " << time4 << " ms" << std::endl;

    cudaFree(d_in_fp32);
    cudaFree(d_out_fp32_1);
    cudaFree(d_out_fp32_2);
    cudaFree(d_in_fp16);
    cudaFree(d_out_fp16_1);
    cudaFree(d_out_fp16_2);

    delete[] h_in_fp32;
    delete[] h_out_fp32_1;
    delete[] h_out_fp32_2;
    delete[] h_in_fp16;
    delete[] h_out_fp16_1;
    delete[] h_out_fp16_2;

    return 0;
}

