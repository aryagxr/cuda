// Tiled 1D convolution

#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 5
#define MAX_MASK_WIDTH 5

__constant__ float M[MAX_MASK_WIDTH];

__global__ void conv1d_tiled(float *N, float *M, float *P, int mask_width, int width){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = mask_width / 2;

    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];

    // left halo cells
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if(threadIdx.x >= blockDim.x - n){
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
    }

    // center halo cells
    N_ds[n + threadIdx.x] = (idx < width) ? N[idx] : 0;

    // right halo cells
    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if(threadIdx.x < n){
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= width) ? 0 : N[halo_index_right];
    }

    __syncthreads();

    float Pval = 0.0f;
    for(int j = 0; j < mask_width; j++){
        Pval += N_ds[threadIdx.x + j] * M[j];
    }

    if(idx < width){
        P[idx] = Pval;
    }
}


int main(){

    const int input_size = 15;
    const int mask_size = 5;

    float N[input_size] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float M[mask_size] = {0.1, 0.2, 0.3, 0.2, 0.1};

    float P[input_size];
    float *d_N, *d_M, *d_P; //device variables

    cudaMalloc((void**)&d_N, input_size*sizeof(float));
    cudaMalloc((void**)&d_M, mask_size*sizeof(float));
    cudaMalloc((void**)&d_P, input_size*sizeof(float)); //output same size as input

    cudaMemcpy(d_N, N, input_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_size*sizeof(float), cudaMemcpyHostToDevice);

    // kernel size 1d
    // 15 threads needed
    int threadsPerBlock = 64;
    int blocksPerGrid = (input_size+threadsPerBlock -1)/threadsPerBlock; // 1 block needed

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    conv1d_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, mask_size, input_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Ensure kernel execution is finished

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    cudaMemcpy(P, d_P, input_size*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Convolution output: ");
    for(int i=0; i<input_size;i++){
        printf("%f ", P[i]);
    }
    

    cudaFree(d_M);cudaFree(d_N);cudaFree(d_P);

    return 0;
}