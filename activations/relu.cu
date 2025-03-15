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






// /* Host helper Functions */

// void initialize_data(float* in_fp32, half* in_fp16, int size) {
//     unsigned long long seed = 1234;
    
//     for (int i = 0; i < size; ++i) {
//         in_fp32[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f; // Random float in range [-1, 1]
//     }
    
//     for (int i = 0; i < size; ++i) {
//         in_fp16[i] = __float2half((rand() / float(RAND_MAX)) * 2.0f - 1.0f); // Random float in range [-1, 1]
//     }
// }

// void print_float_array(float* arr, int size) {
//     for (int i = 0; i < size; ++i) {
//         std::cout << arr[i] << " ";
//     }
//     std::cout << std::endl;
// }

// void print_half_array(half* arr, int size) {
//     for (int i = 0; i < size; ++i) {
//         std::cout << __half2float(arr[i]) << " ";
//     }
//     std::cout << std::endl;
// }


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
    

    kernel1_relu_fp32<<<blocksPerGrid, threadsPerBlock>>>(dx, dp, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Scalar FP32 Kernel1 execution time: " << ms << " ms\n";

    cudaMemcpy(P, dp, fp32_size, cudaMemcpyDeviceToHost);


    // Run vectorized FP32 * 4 kernel
    cudaEventRecord(start);
    kernel2_relu_4fp32<<<blocksPerGrid_vec, threadsPerBlock>>>(dx, dp2, N);
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

