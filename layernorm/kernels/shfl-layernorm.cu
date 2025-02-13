#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

/*  This is an optimized implementation of layer normalization.
    One block processes a whole row of the input matrix.
    Warp level shuffle functions are used to perform sum reduction.
*/

#define EPSILON 1e-3

__global__ void shfl_layernorm(float *X, float *P, int m, int n){

    __shared__ float smem[1024];
    
    int row = blockIdx.x; //one block per row
    int tidx = threadIdx.x;
    int warp_size = 32;

    if(row < m){

        float *row_in = X + row * n;
        float *row_out = P + row * n;

        float lmean = 0.0f;
        float lvar = 0.0f;

        // local mean
        for(int i=tidx; i<n; i += blockDim.x){
            float a = row_in[i];
            lmean += a;
            lvar += a*a;
        }

        __syncthreads();

        // store in register instead of smem
        float lrmean = lmean;

        // global mean, warp level using shuffling
        for(int offset = warp_size/2; offset > 0; offset /= 2){
            lrmean += __shfl_down_sync(0xffffffff, lrmean, offset);
        }
        // at this point, each warp finished summing values at each warp
        // sum of each warp is stored at 0 index of each warp


        // global mean, block level using shuffling
        if (blockDim.x > warp_size){
            if(tidx % warp_size == 0){
                smem[tidx/warp_size] = lrmean; //store sum of each warp into smem
            }
            __syncthreads();

            if(tidx < warp_size){ // only first warp
                lrmean = (tidx < (blockDim.x + warp_size - 1) / warp_size) ? smem[tidx] : 0.0f;
                for(int offset = warp_size / 2; offset > 0; offset /=2){
                    lrmean = __shfl_down_sync(0xffffffff, lrmean, offset);
                }

                if(tidx==0){
                    smem[0] = lrmean;
                }
            }
        }
        else{
            if(tidx==0){
                smem[0] = lrmean;
            }
        }
        __syncthreads();

        float gmean = smem[0] / n;
        __syncthreads();

        if (tidx == 0) {
            printf("Row: %d, Local Mean: %f, Global Mean: %f\n", row, lmean, gmean);
        }


        
        // local variance
        float lrvar = lvar;

        if (tidx == 0) {
            printf("Row: %d, Local Variance: %f\n", row, lrvar);
        }

        for(int offset = warp_size/2; offset > 0; offset /= 2){
            lrvar += __shfl_down_sync(0xffffffff, lrvar, offset);
        }

        // block level variance
        if (blockDim.x > warp_size){
            if(tidx % warp_size == 0){
                smem[tidx/warp_size] = lrvar;
            }
            __syncthreads();

            if(tidx < warp_size){
                lrvar = (tidx < (blockDim.x + warp_size - 1) / warp_size) ? smem[tidx] : 0.0f;
                for(int offset = warp_size / 2; offset > 0; offset /=2){
                    lrvar += __shfl_down_sync(0xffffffff, lrvar, offset);
                }

                if(tidx == 0){
                    smem[0] = lrvar;
                }
            }
        }
        else{
            if(tidx == 0){
                smem[0] = lrvar;
            }
        }
        __syncthreads();

        float gvar = (smem[0] / n) -  (gmean * gmean); // Load global variance

        if (tidx == 0) {
            printf("Row: %d, Global Variance: %f\n", row, gvar);
        }

        gvar = fmaxf(gvar, 0.0f);
        float stddev = rsqrtf(gvar + EPSILON); // 1/stddev
        __syncthreads();

        for(int i=tidx; i<n; i += blockDim.x){
            row_out[i] = (row_in[i] - gmean) * stddev;
        }
    }
    else{
        return;
    }
}

void run_shfl_ln(float *D_in, float *D_out, int m, int n){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(m);
    //dim3 blocksPerGrid(ceil(m/threadsPerBlock.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    shfl_layernorm<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}