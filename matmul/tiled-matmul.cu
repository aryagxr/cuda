// tiled matrix multiplication

#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define N 1024

__global__  void tiledMatMul(float *d_M, float *d_N, float *d_P, int width){
    
    // shared variables
    __shared__ float M_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_shared[TILE_WIDTH][TILE_WIDTH];

    // automatic variables saved into registers
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // index
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    // automatic variable
    float P = 0;

    // iterate through phases to compute P
    int max_phases = width / TILE_WIDTH;
    for (int phase = 0; phase < max_phases; ++phase){

        // load each tile element into shared memory
        M_shared[ty][tx] = d_M[row*width + phase*TILE_WIDTH + tx];
        N_shared[ty][tx] = d_N[(phase*TILE_WIDTH + ty) * width + col];

        __syncthreads(); // wait for all threads to finish loading tiles

        // each thread computes an element of the output matrix
        for (int k = 0; k < TILE_WIDTH; ++k){
            P += M_shared[ty][k] * N_shared[k][tx]; 
        }
        __syncthreads();
    }

    d_P[row*width + col] = P;
}


int main(){

    size_t matrix_size = N*N*sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(matrix_size);
    h_B = (float*)malloc(matrix_size);
    h_C = (float*)malloc(matrix_size);

    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    for(int i=0; i<N*N; i++){
        h_A[i] = (i % N) + 1;
        h_B[i] = (i / N) + 1;
    }

    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(N/TILE_WIDTH, N/TILE_WIDTH);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    tiledMatMul<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Ensure kernel execution is finished

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";


    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);

    printf("Successful");
    printf("Sample result C[0][0] = %f\n", h_C[0]);

    // free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}