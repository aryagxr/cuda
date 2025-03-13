#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#define N 1024
#define NSTREAMS 4

__global__ void vect_add_streams(float* a, float* b, float* c, int offset, int n){

    //int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if(tidx < offset + n){
        c[tidx] = a[tidx] + b[tidx];
    }

}


int main(){
    
    int size = N * sizeof(float);
    const int threadsPerBlock = 256;
    const int streamSize = N / NSTREAMS;
    int streamBytes = streamSize * sizeof(float);
    int blocksPerGrid = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

    // Using pinned host memory
    float *ha, *hb, *hc;
    cudaMallocHost(&ha, size);
    cudaMallocHost(&hb, size);
    cudaMallocHost(&hc, size);

    float *da, *db, *dc;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);

    for (int i = 0; i < N; i++) {
        ha[i] = static_cast<float>(i);
        hb[i] = static_cast<float>(i * 2);
    }

    std::cout << "First 10 elements of A: ";
    for (int i = 0; i < 10; i++) {
        std::cout << ha[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "First 10 elements of B: ";
    for (int i = 0; i < 10; i++) {
        std::cout << hb[i] << " ";
    }
    std::cout << std::endl;



    cudaStream_t stream[NSTREAMS];
    for(int i = 0; i < NSTREAMS; i++){
        cudaStreamCreate(&stream[i]);
    }


    /* Async Version 1: Copy, Kernel, Copy per stream*/
    for(int i = 0; i < NSTREAMS; i++){
        int offset = streamSize * i;

        // copy: H2D
        cudaMemcpyAsync(&da[offset], &ha[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&db[offset], &hb[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);

        // kernel
        vect_add_streams<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(da, db, dc, offset, streamSize);

        // copy: D2H
        cudaMemcpyAsync(&hc[offset], &dc[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);

    }

    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    std::cout << "First 10 elements of C: ";
    for (int i = 0; i < 10; i++) {
        std::cout << hc[i] << " ";
    }
    std::cout << std::endl;

    bool success = true;
    for (int i = 0; i < N; i++) {
        if (hc[i] != ha[i] + hb[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << std::endl;
            break;
        }
    }
    std::cout << (success ? "Success!" : "Failed!") << std::endl;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFreeHost(ha);
    cudaFreeHost(hb);
    cudaFreeHost(hc);

    return 0;

}
