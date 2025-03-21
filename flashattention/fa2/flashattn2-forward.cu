#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void fa2_forward(float *Q, float *K, float*V, 
                            float *O, float *l, float *m, 
                            int N, int d, float scale,
                            int Br, int Bc, int Tr, int Tc){

    int tx = threadIdx.x;
    int bx = blockIdx.x; // Batch size
    int by = blockIdx.y; // number of heads


    // global mem offsets
    // (bx * gridDim.y * N * d) -> current batch
    // (by * N * d) -> current head within the batch
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N * d) + (by * N);

    int kv_tile = Bc * d;
    int qo_tile = Br * d;
    int Si_tile = Br * Bc;
    int lm_tile = Br;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = &smem[qo_tile];
    float* Vj = &smem[qo_tile + kv_tile];
    float* Si = &smem[qo_tile + (kv_tile * 2)];
    float* Oi = &smem[qo_tile + (kv_tile * 2) + Si_tile];
    float* li = &smem[(qo_tile * 2) + (kv_tile * 2) + Si_tile];
    float* mi = &smem[(qo_tile * 2) + (kv_tile * 2) + Si_tile  + lm_tile];



    // 3: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do
    // iterate through blocks of Q
    for(int i = 0; i < Tr; i++){

        // 4: Load Q𝑖 from HBM to on-chip SRAM
        // one thread loads one row of Qi block
        for(int x = 0; x < d; x++){
            Qi[x + (tx * d)] = Q[qkv_offset + (qo_tile * i) + (tx * d) + x];
            Oi[x + (tx * d)] = 0.0f; // 5: Oi(0) = (0)Br*d;
        }
        __syncthreads();

        // 5: On chip initialize li(0) = (0)Br; mi(0) = (-inf)Br
        if(tx < Br){
            mi[tx] = -INFINITY;
            li[tx] = 0.0f;
        }

        // 6: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do
        for(int j = 0; j < Tc; j++){

            // 7: Load K𝑗, V𝑗 from HBM to on-chip SRAM
            for(int y = 0; y < d; y++){
                Kj[y + (tx * d)] = K[qkv_offset + (kv_tile * j) + (tx * d) + y];
                Vj[y + (tx * d)] = V[qkv_offset + (kv_tile * j) + (tx * d) + y];
            }

        }


    }
    
}


int main(){

    int BATCH = 2;
    int HEADS = 2;
    int SEQ_LEN = 3;
    int EMBED_DIM = 3;
    int total_size = BATCH * HEADS * SEQ_LEN * EMBED_DIM;
    int Br = SEQ_LEN;
    int Bc = SEQ_LEN;
    int Tr = 1;
    int Tc = 1;

    // Allocate host memory
    float *h_Q = new float[total_size];
    float *h_K = new float[total_size];
    float *h_V = new float[total_size];
    float *h_m = new float[BATCH * HEADS * SEQ_LEN];
    float *h_l = new float[BATCH * HEADS * SEQ_LEN];
    float *h_O = new float[total_size];
}