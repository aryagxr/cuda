#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void kogglestone(float* __restrict__ X, float* __restrict__ Y, int n){

    // shared memory
    __shared__ float smem[BLOCK_SIZE];

    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < n){
        smem[threadIdx.x] = X[tidx];
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if(threadIdx.x >= stride){
            smem[threadIdx.x] += smem[threadIdx.x - stride];
        }
    }

    Y[tidx] = smem[threadIdx.x];

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

    kogglestone<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, ARRAY_SIZE);
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