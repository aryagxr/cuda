#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#define N 1024
#define NSTREAMS 4


/* 1 block per stream */
__global__ void relu_streams(float* in, float* out, int n, int offset){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x + offset;

    if(tidx < offset + n){
        out[tidx] = fmaxf(0.0, in[tidx]);
    }
    __syncthreads();
}



int main(){

    size_t vec_size = N * sizeof(float);
    int streamSize = N / NSTREAMS;
    int streamBytes = streamSize * sizeof(float);

    int threads = 256;
    int blocks = (streamSize + threads - 1) / threads;

    // pinned memory
    float *hx, *hp;
    cudaMallocHost(&hx, vec_size);
    cudaMallocHost(&hp, vec_size);
    float *dx, *dp;
    cudaMalloc((void**)&dx, vec_size);
    cudaMalloc((void**)&dp, vec_size);

    for (int i = 0; i < N; ++i) {
        hx[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }

    std::cout << "First 10 elements of Input: ";
    for (int i = 0; i < 10; i++) {
        std::cout << hx[i] << " ";
    }
    std::cout << std::endl;

    cudaStream_t stream[NSTREAMS];
    for(int i = 0; i < NSTREAMS; i++){
        cudaStreamCreate(&stream[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    /* Async Version 1: Copy, Kernel, Copy per stream*/
    for(int i = 0; i < NSTREAMS; i++){

        int offset = i * streamSize;

        // H2D
        cudaMemcpyAsync(&dx[offset], &hx[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);

        // kernel
        relu_streams<<<blocks, threads, 0, stream[i]>>>(dx, dp, streamSize, offset);


        // D2H
        cudaMemcpyAsync(&hp[offset], &dp[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);

    }

    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Time taken for kernel execution: " << elapsedTime << " ms" << std::endl;


    std::cout << "First 10 elements of Output: ";
    for (int i = 0; i < 10; i++) {
        std::cout << hp[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(dx); cudaFree(dp);
    cudaFreeHost(hx); cudaFreeHost(hp);

    return 0;
}