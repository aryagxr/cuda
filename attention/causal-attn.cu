#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define NUM_WORDS 6
#define EMBED_DIM 3
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NEG_INF -1e9f


struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
    __device__ __forceinline__ float identity() const { return 0.0f; }
};

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
    __device__ __forceinline__ float identity() const { return -INFINITY; }
};


template <typename Op>
__device__ __forceinline__ float warpReduce(float val, Op op) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = op(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

template <typename Op>
__device__ __forceinline__ void blockReduce(float& val, float* smem, int tidx, int threadsPerBlock, Op op) {
    val = warpReduce(val, op);

    if (tidx % WARP_SIZE == 0) {
        smem[tidx / WARP_SIZE] = val;
    }
    __syncthreads();

    if (tidx < WARP_SIZE) {
        val = (tidx < (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE) ? smem[tidx] : op.identity();
        val = warpReduce(val, op);
        if (tidx == 0) {
            smem[0] = val;
        }
        
    }
    __syncthreads();
}

// attention scores with causal masking
__global__ void attn_scores(float* __restrict__ input, float* __restrict__ attn_scores) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= NUM_WORDS) return;

    float dot_prod = 0.0f;
    #pragma unroll
    for (int i = 0; i < EMBED_DIM; i++) {
        dot_prod += input[row * EMBED_DIM + i] * input[col * EMBED_DIM + i];
    }

    // Apply causal masking: Zero out upper triangular part
    if (col > row) dot_prod = NEG_INF;

    attn_scores[row * NUM_WORDS + col] = dot_prod;
}

// softmax with masking applied
__global__ void softmax_inplace(float* __restrict__ attn_scores) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if (row >= NUM_WORDS) return;
    float* row_ptr = attn_scores + row * NUM_WORDS;

    float lmax = row_ptr[tidx];
    blockReduce(lmax, smem, tidx, NUM_WORDS, MaxOp());
    float gmax = smem[0];

    float lnorm = (row_ptr[tidx] == NEG_INF) ? 0.0f : expf(row_ptr[tidx] - gmax);
    blockReduce(lnorm, smem, tidx, NUM_WORDS, SumOp());
    float gnorm = smem[0];

    row_ptr[tidx] = (row_ptr[tidx] == NEG_INF) ? 0.0f : expf(row_ptr[tidx] - gmax) / gnorm;
}

// Compute context vectors
__global__ void context_vector(float* __restrict__ input, float* __restrict__ attn_weights, float* __restrict__ context_vect) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    float context = 0.0f;
    if (col >= EMBED_DIM) return;

    #pragma unroll
    for (int i = 0; i <= row; i++) {  // Attend only to itself and previous tokens
        context += input[i * EMBED_DIM + col] * attn_weights[row * NUM_WORDS + i];
    }
    context_vect[row * EMBED_DIM + col] = context;
}

int main() {
    float token_emb[NUM_WORDS * EMBED_DIM] = {
        0.43, 0.15, 0.89,
        0.55, 0.87, 0.66,
        0.57, 0.85, 0.64,
        0.22, 0.58, 0.33,
        0.77, 0.25, 0.10,
        0.05, 0.80, 0.55
    };

    float *d_inputs, *d_attention_scores, *d_context_vectors;
    cudaMalloc(&d_inputs, NUM_WORDS * EMBED_DIM * sizeof(float));
    cudaMalloc(&d_attention_scores, NUM_WORDS * NUM_WORDS * sizeof(float));
    cudaMalloc(&d_context_vectors, NUM_WORDS * EMBED_DIM * sizeof(float));

    cudaMemcpy(d_inputs, token_emb, NUM_WORDS * EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice);

    attn_scores<<<NUM_WORDS, NUM_WORDS>>>(d_inputs, d_attention_scores);
    cudaDeviceSynchronize();

    float h_attention_scores[NUM_WORDS * NUM_WORDS];
    cudaMemcpy(h_attention_scores, d_attention_scores, NUM_WORDS * NUM_WORDS * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Attention Scores Before Softmax (Causal Mask Applied):\n";
    for (int i = 0; i < NUM_WORDS; i++) {
        for (int j = 0; j < NUM_WORDS; j++) {
            std::cout << h_attention_scores[i * NUM_WORDS + j] << " ";
        }
        std::cout << "\n";
    }

    softmax_inplace<<<NUM_WORDS, NUM_WORDS, NUM_WORDS * sizeof(float)>>>(d_attention_scores);
    cudaDeviceSynchronize();

    cudaMemcpy(h_attention_scores, d_attention_scores, NUM_WORDS * NUM_WORDS * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Attention Weights After Softmax (Causal Mask Applied):\n";
    for (int i = 0; i < NUM_WORDS; i++) {
        for (int j = 0; j < NUM_WORDS; j++) {
            std::cout << h_attention_scores[i * NUM_WORDS + j] << " ";
        }
        std::cout << "\n";
    }

    context_vector<<<NUM_WORDS, EMBED_DIM>>>(d_inputs, d_attention_scores, d_context_vectors);
    cudaDeviceSynchronize();

    float h_context_vectors[NUM_WORDS * EMBED_DIM];
    cudaMemcpy(h_context_vectors, d_context_vectors, NUM_WORDS * EMBED_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Context Vectors:\n";
    for (int i = 0; i < NUM_WORDS; i++) {
        std::cout << "(";
        for (int d = 0; d < EMBED_DIM; d++) {
            std::cout << h_context_vectors[i * EMBED_DIM + d] << (d < EMBED_DIM - 1 ? ", " : ")\n");
        }
    }

    cudaFree(d_inputs);
    cudaFree(d_attention_scores);
    cudaFree(d_context_vectors);

    return 0;
}