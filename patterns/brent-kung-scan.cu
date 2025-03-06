#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define N 1024

__global__ void brent_kung_scan(float* __restrict__ X, float* __restrict__ Y, int n){

    __shared__ float smem[BLOCK_SIZE];

    int i = (2 * blockDim.x * blockIdx.x) + threadIdx.x;
    if(i < n){
        smem[threadIdx.x] = X[i];
    }
    if(i+blockDim.x < n){
        smem[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    }

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();

        int idx = (threadIdx.x + 1) * 2 * stride -1;
        if(idx < BLOCK_SIZE){
            smem[idx] += smem[idx - stride];
        }
    }

    for(int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2){
        __syncthreads();
        int idx = (threadIdx.x + 1) * stride*2 - 1;
        if(idx + stride < BLOCK_SIZE){
            smem[idx + stride] += smem[idx];
        }
    }

    __syncthreads();

    if(i < n){
        Y[i] = smem[threadIdx.x];
    }

    if(i + blockDim.x < n){
        Y[i + blockDim.x] = smem[threadIdx.x + blockDim.x];
    }
}


int main() {
    const int ARRAY_SIZE = N; 
    const int SIZE = ARRAY_SIZE * sizeof(float);
    
    float h_X[ARRAY_SIZE]; 
    float h_Y[ARRAY_SIZE]; 
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_X[i] = i + 1;
    }

    float *d_X, *d_Y;

    cudaMalloc((void**)&d_X, SIZE);
    cudaMalloc((void**)&d_Y, SIZE);

    cudaMemcpy(d_X, h_X, SIZE, cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE; 
    int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    brent_kung_scan<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, ARRAY_SIZE);
    cudaMemcpy(h_Y, d_Y, SIZE, cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum Results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_Y[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}