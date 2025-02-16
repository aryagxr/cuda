// Tiled 1D convolution using general cache

#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 5
#define MAX_MASK_WIDTH 5

__global__ void conv1d_cache(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE];
    
    N_ds[threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    
    int cur_tile_start_point = blockIdx.x * blockDim.x;
    int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Mask_Width / 2);
    float Pvalue = 0;
    
    for (int j = 0; j < Mask_Width; j++) {
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index < Width) {
            if ((N_index >= cur_tile_start_point) && (N_index < next_tile_start_point)) {
                Pvalue += N_ds[threadIdx.x + j - (Mask_Width / 2)] * M[j];
            } else {
                Pvalue += N[N_index] * M[j];
            }
        }
    }
    P[i] = Pvalue;
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

    conv1d_cache<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, mask_size, input_size);
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