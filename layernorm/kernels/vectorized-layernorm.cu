#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#define EPSILON 1e-6
#define WARP_SIZE 32

__global__ void vectorized_layernorm(float *X, float *P, int m, int n){

    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if(row >= m) return;

    float *row_in = X + row * n;
    float *row_out = P + row * n;
    float lmean = 0.0f;
    float lvar = 0.0f;
    
    int vec_iters = n / 4;
    // int rem = n % 4;

    // summing up the 4 elements loaded by each thread
    for(int i = tidx; i < vec_iters; i+= blockDim.x){
        float4 v = reinterpret_cast<float4 *>(row_in)[i];
        lmean += v.x + v.y + v.z + v.w;
        lvar += (v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w);
    }

    // reducing sum across warps
    for(int offset = WARP_SIZE / 2; offset > 0; offset /=2){
        lmean += __shfl_down_sync(0xffffffff, lmean, offset);
        lvar += __shfl_down_sync(0xffffffff, lvar, offset);
    }

    float gmean = lmean / n;
    float gvar = (lvar / n) - (gmean * gmean);
    float std_inv = rsqrtf(gvar + EPSILON);

    for(int i = tidx; i < vec_iters; i += blockDim.x){
        float4 v = reinterpret_cast<float4 *>(row_in)[i];
        v.x = (v.x - gmean) * std_inv;
        v.y = (v.y - gmean) * std_inv;
        v.z = (v.z - gmean) * std_inv;
        v.w = (v.w - gmean) * std_inv;
        reinterpret_cast<float4 *>(row_out)[i] = v;
    }

    // remainder elements not included in vectors
    for(int i = vec_iters * 4 + tidx; i < n; i += blockDim.x){
        row_out[i] = (row_in[i] - gmean) * std_inv;
    }


}


void run_vect_ln(float *D_in, float *D_out, int m, int n){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(m);
    //dim3 blocksPerGrid(ceil(m/threadsPerBlock.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    vectorized_layernorm<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}