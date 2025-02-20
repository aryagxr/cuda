#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


// macros
#define NUM_WORDS 6
#define EMBED_DIM 3
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff


// cuda error check


// operator overloading for max and sum
struct SumOp{
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
    __device__ __forceinline__ float identity() const { return 0.0f; }
};

struct MaxOp{
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a,b);}
    __device__ __forceinline__ float identity() const { return -INFINITY; }
};



// device functions
template <typename Op>
__device__ __forceinline__ float warpReduce(float val, Op op){
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2){
        val = op(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

template <typename Op>
__device__ __forceinline__ void blockReduce(float val, float *smem, int tidx, int threadsPerBlock, Op op){
    
    val = warpReduce(val, op);

    if(tidx % WARP_SIZE == 0){
        smem[tidx/WARP_SIZE] = val;
    }
    __syncthreads();

    if(tidx < WARP_SIZE){
        val = (tidx < threadsPerBlock / WARP_SIZE) ? smem[tidx] : op.identity();
        val = warpReduce(val, op);
        if (tidx == 0) smem[0] = val;
    }
    __syncthreads();
}



__global__ void attn_scores(float* __restrict__ input, float* __restrict__ attn_scores){

    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= NUM_WORDS) return;

    float dot_prod = 0.0f;

    #pragma unroll
    for(int i = 0; i < EMBED_DIM; i++){
        dot_prod += input[row * EMBED_DIM + i] * input[col * EMBED_DIM + i];
    }

    attn_scores[row * NUM_WORDS + col] = dot_prod;
}


__global__ void softmax_inplace(float* __restrict__ attn_scores){
    
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if(row >= NUM_WORDS) return;
    float *row_ptr = attn_scores + row * NUM_WORDS;

    float lmax = row_ptr[tidx];
    blockReduce(lmax, smem, tidx, NUM_WORDS, MaxOp());
    float gmax = smem[0];

    float lnorm = expf(row_ptr[tidx] - gmax);
    blockReduce(lnorm, smem, tidx, NUM_WORDS, SumOp());
    float gnorm = smem[0];

    row_ptr[tidx] = expf(row_ptr[tidx] - gmax) / gnorm;

}


