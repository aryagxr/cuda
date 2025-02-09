#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#define M 1024
#define N 32768

// sotfmax using warp shuffle functions

__global__ void shfl_softmax(float *X, float *P, int m, int n){

    __shared__ float smem[1024];

    // one block for one row
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    unsigned int warp_size = 32;


    if(row >= m){
        return;
    }

    // load from global memory to registers
    float *row_in = X + row * n;
    float *row_out = P + row * n;

    float lmax = -1 * INFINITY;
    float lnorm = 0.0f;

    // computing local max and local norm
    for(int i = tidx; i<n; i += blockDim.x){
        float x = row_in[i]; // from global mem to register

        if(x > lmax){
            lnorm *= expf(lmax-x);
            lmax = x;
        }
        lnorm += expf(x-lmax);
    }

    __syncthreads();

    // warp reduction gets local max value from registers 
    // registers faster than smem
    // warp level reduction
    float lrmax = lmax;
    for(int offset = warp_size/2; offset > 0; offset /=2){
        lrmax = fmaxf(lrmax, __shfl_down_sync(0xffffffff, lrmax, offset));
    }


    // block level reduction
    // if a block only has 32 threads, warp level reduction is enough
    // but if block dim is greater than 32, block level reduction needed
    if(blockDim.x > warp_size){
        if(tidx % warp_size == 0){ //if beginning of warp
            smem[tidx/warp_size] = lrmax; //store warp max in smem
        }
        __syncthreads();

        if(tidx < warp_size){
            lrmax = (tidx < (blockDim.x + warp_size - 1) / warp_size) ? smem[tidx] : -INFINITY;
            for(int offset = warp_size / 2; offset > 0; offset /=2){
                lrmax = fmaxf(lrmax, __shfl_down_sync(0xffffffff, lrmax, offset));
            }

            if(tidx == 0){
                smem[0] = lrmax; // final max stored in smem[0]
            }
        }

    }
    // if blockdim < warp_size
    else{
        if(tidx == 0){
            smem[0] = lrmax;
        }
    }
    __syncthreads();

    float gmax = smem[0];
    __syncthreads();


    // local norm
    float lrnorm = lnorm;
    lrnorm = lnorm * expf(lmax - gmax);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        lrnorm += __shfl_down_sync(0xffffffff, lrnorm, offset);
    }
    if(blockDim.x > warp_size){
        if(tidx % warp_size == 0){ //if beginning of warp
            smem[tidx/warp_size] = lrnorm; //store warp max in smem
        }
        __syncthreads();

        if(tidx < warp_size){
            lrnorm = (tidx < (blockDim.x + warp_size - 1) / warp_size) ? smem[tidx] : -INFINITY;
            for(int offset = warp_size / 2; offset > 0; offset /=2){
                lrnorm = fmaxf(lrnorm, __shfl_down_sync(0xffffffff, lrnorm, offset));
            }

            if(tidx == 0){
                smem[0] = lrnorm; // final max stored in smem[0]
            }
        }

    }
    // if blockdim < warp_size
    else{
        if(tidx == 0){
            smem[0] = lrnorm;
        }
    }
    __syncthreads();

    float gnorm = smem[0];


    // putting max and norm together, compute softmax
    for (int i = tidx; i < n; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - gmax) / gnorm;
    }


}


int main(){

    size_t matrix_size = M*N*sizeof(float);

    float *X_input, *P_output;
    float *D_input, *D_output;

    X_input = (float*)malloc(matrix_size);
    P_output = (float*)malloc(matrix_size);

    cudaMalloc((void**)&D_input, matrix_size);
    cudaMalloc((void**)&D_output, matrix_size);

    for(int a = 0; a<M*N; a++){
        X_input[a] = ((float)rand() / RAND_MAX) * 10.0f;
    }

    cudaMemcpy(D_input, X_input, matrix_size, cudaMemcpyHostToDevice);


    
    dim3 ThreadsPerBlock(1024);
    dim3 blocksPerGrid(M); // m rows, m blocks

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    shfl_softmax<<<blocksPerGrid, ThreadsPerBlock>>>(D_input, D_output, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(P_output, D_output, matrix_size, cudaMemcpyDeviceToHost);

    printf("Softmax result (first row only):\n");
    for (int j = 0; j < 10; j++) { 
        printf("%f ", P_output[j]);
    }
    printf("...\n");

    cudaFree(D_input); cudaFree(D_output);
    free(X_input); free(P_output);

    return 0;

}