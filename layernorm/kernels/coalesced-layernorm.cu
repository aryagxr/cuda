#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


#define EPSILON 1e-6


/*  This kernel uses memory coalescing,
    by storing input matrix as column major.
*/

__global__ void coalesced_layernorm(float *X, float *P, int m, int n){

    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if(col >= n) return;

    float mean = 0.0f;
    float var = 0.0f;

    for(int row = 0; row < m; row++){

        // column major
        int idx = col * m + row;
        mean += X[idx];
    }
    mean /= m;

    for(int row = 0; row < m; row++){

        int idx = col * m + row;
        var += (X[idx] - mean) * (X[idx] - mean);
    }
    var /= m;

    float stddev = sqrtf(var + EPSILON);
    for(int row = 0; row < m; row++){
        int idx = col * m + row;
        P[idx] = (X[idx] - mean) / stddev;
    }

}

void run_coalesced_ln(float *D_in, float *D_out, int m, int n){

    dim3 threadsPerBlock(1024); // 1024 rows
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
    //dim3 blocksPerGrid(ceil(m/threadsPerBlock.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    coalesced_layernorm<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}

