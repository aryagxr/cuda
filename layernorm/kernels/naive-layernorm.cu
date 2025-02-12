#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


#define EPSILON 1e-6


/*  This is a naive implementation of layer normalization.
    One thread processes a whole row of the input matrix.
*/


__global__ void naive_layernorm(float *X, float *P, int m, int n){

    int row = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < m){

        float mean = 0.0;
        float var = 0.0;


        // find mean of that row
        for(int col = 0; col < n; col++){
            int idx = row * n + col;
            mean += X[idx];    
        }
        mean /= n;

        // compute variance
        for(int col = 0; col < n; col++){
            int idx = row * n + col;
            var += (X[idx] - mean) * (X[idx] - mean);
        }
        var /= n;

        // normalize each row
        float stddev = sqrt(var + EPSILON);
        for(int col = 0; col < n; col++){
            int idx = row * n + col;
            P[idx] = (X[idx] - mean) / stddev;
        }


    }
    else{
        return;
    }

}



void run_naive_ln(float *D_in, float *D_out, int m, int n){

    dim3 threadsPerBlock(1024); // 1024 rows
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
    //dim3 blocksPerGrid(ceil(m/threadsPerBlock.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    naive_layernorm<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}

