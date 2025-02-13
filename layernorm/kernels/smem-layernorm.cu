#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define EPSILON 1e-6

__global__ void smem_layernorm(float *X, float *P, int m, int n){

    __shared__ float smem[1024];
    
    int row = blockIdx.x; //one block per row
    int tidx = threadIdx.x;
    

    if(row < m){

        float *row_in = X + row * n;
        float *row_out = P + row * n;

        float lmean = 0.0f;
        float lvar = 0.0f;

        // local mean and var
        for(int i = tidx; i < n; i+=blockDim.x){
            float a = row_in[i]; // load from global mem into register
            lmean += a;
            lvar += (a*a);
        }

        __syncthreads();
        smem[tidx] = lmean; // contains the sum of all values loaded by that thread tidx
        __syncthreads();

        // global mean: using reduction
        for(int stride = blockDim.x/2; stride >0; stride /=2){
            if(tidx < stride){
                smem[tidx] += smem[tidx + stride]; // the sum will end up in index 0
            }
            __syncthreads();
        }

        float gmean = smem[0] / n;
        __syncthreads();

        // now we can store local squared sums in smem
        smem[tidx] = lvar;
        __syncthreads();

        // global variance: using reduction
        for(int stride = blockDim.x; stride > 0; stride /= 2){
            if(tidx < stride){
                smem[tidx] += smem[tidx + stride];

            }
            __syncthreads();
        }
        float gvar = (smem[0]/n) - (gmean * gmean);
        float stddev = rsqrtf(gvar + EPSILON); // 1/stddev
        __syncthreads();

        // normalize and store outputs
        for(int i = tidx; i < n; i += blockDim.x){
            row_out[i] = (row_in[i] - gmean) * stddev;
        }
    }
    else{
        return;
    } 
}


void run_smem_ln(float *D_in, float *D_out, int m, int n){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(m);
    //dim3 blocksPerGrid(ceil(m/threadsPerBlock.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    smem_layernorm<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}