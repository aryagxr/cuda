#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define M 1024 //rows
#define N 1024 //columns


/* Kernel 1: Half precision matvec multiplication.
   C = A * v
   C is the accumulator, A is a matrix, v is a vector.
   One thread per row.
   __hadd 
*/

__global__ void kernel1_hgemv_fp16(half* A, half* v, half* C){

    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    if(row > M) return;

    half acc = __float2half(0.0f);
    for(int i = 0; i < N; i++){
        acc = __hadd(acc, __hmul(A[row * N + i], v[i]));
    }
    C[row] = acc;

}


void run_kernel1_hgemv_fp16(half* d_A, half *d_v, half* d_C){
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;

    kernel1_hgemv_fp16<<<gridSize, blockSize>>>(d_A, d_v, d_C);
}