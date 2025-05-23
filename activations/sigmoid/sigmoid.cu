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

__global__ void kernel3_sigmoid_fp16(half* in, half* out, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < n) {
        half x = in[tidx];
        float x_float = __half2float(x);
        float result_float = 1.0f / (1.0f + expf(-x_float));
        out[tidx] = __float2half(result_float);
    }
}



__global__ void kernel4_sigmoid_4fp16(half* in, half* out, int n) {
    int tidx = (threadIdx.x + (blockDim.x * blockIdx.x)) * 4;
    
    if(tidx < n) {
        half2* in2 = &HALF2(in[tidx]);
        half2* out2 = &HALF2(out[tidx]);
        
        half2 x1 = in2[0]; 
        half2 x2 = in2[1];

        float2 fx1 = __half22float2(x1);
        float2 fx2 = __half22float2(x2);
        

        float2 fy1, fy2;
        fy1.x = 1.0f / (1.0f + expf(-fx1.x));
        fy1.y = 1.0f / (1.0f + expf(-fx1.y));
        fy2.x = 1.0f / (1.0f + expf(-fx2.x));
        fy2.y = 1.0f / (1.0f + expf(-fx2.y));

        half2 y1 = __float22half2_rn(fy1);
        half2 y2 = __float22half2_rn(fy2);

        out2[0] = y1;
        out2[1] = y2;
    }
}




int main(){
    const int N = 1024;
    size_t fp32_size = N * sizeof(float);
    size_t fp16_size = N * sizeof(half);

    float *X, *P, *P2;
    float *dx, *dp, *dp2;
    half *X_half;
    float *P3;
    half *dx_half, *dp_half;
    half *dp_half_vec;
    
    X = (float*)malloc(fp32_size);
    P = (float*)malloc(fp32_size);
    P2 = (float*)malloc(fp32_size);
    P3 = (float*)malloc(fp32_size);
    X_half = (half*)malloc(fp16_size);

    cudaMalloc((void**)&dx, fp32_size);
    cudaMalloc((void**)&dp, fp32_size);
    cudaMalloc((void**)&dp2, fp32_size);
    cudaMalloc((void**)&dx_half, fp16_size);
    cudaMalloc((void**)&dp_half, fp16_size);
    cudaMalloc((void**)&dp_half_vec, fp16_size);
    
    for (int i = 0; i < N; ++i) {
        X[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
        X_half[i] = __float2half(X[i]);
    }


    cudaMemcpy(dx, X, fp32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dx_half, X_half, fp16_size, cudaMemcpyHostToDevice);

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


    cudaEventRecord(start);
    kernel3_sigmoid_fp16<<<blocksPerGrid, threadsPerBlock>>>(dx_half, dp_half, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "FP16 Kernel3 execution time: " << ms << " ms\n";



    cudaEventRecord(start);
    kernel4_sigmoid_4fp16<<<blocksPerGrid_vec, threadsPerBlock>>>(dx_half, dp_half_vec, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Vectorized FP16*4 Kernel4 execution time: " << ms << " ms\n";
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(P2, dp2, fp32_size, cudaMemcpyDeviceToHost);
    half* P_half = (half*)malloc(fp16_size);
    cudaMemcpy(P_half, dp_half, fp16_size, cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < N; ++i) {
        P3[i] = __half2float(P_half[i]);
    }


    for (int i = 0; i < 10; ++i) {
        std::cout << P[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << P2[i] << " ";
    }

    std::cout << "\n\nFirst 10 results from FP16 kernel:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << P3[i] << " ";
    }
    std::cout << std::endl;

    half* P_half_vec = (half*)malloc(fp16_size);
    float* P4 = (float*)malloc(fp32_size);
    cudaMemcpy(P_half_vec, dp_half_vec, fp16_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        P4[i] = __half2float(P_half_vec[i]);
    }

    std::cout << "\nFirst 10 results from FP16 vectorized kernel:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << P4[i] << " ";
    }
    std::cout << std::endl;


    cudaFree(dx); cudaFree(dp); cudaFree(dp2); cudaFree(dx_half); cudaFree(dp_half);
    free(X); free(P); free(P2); free(P3); free(X_half); free(P_half);
    cudaFree(dp_half_vec);
    free(P_half_vec);
    free(P4);


    return 0;
    
}



