#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>





__global__ void fa1_forward(float *Q, float *K, float*V, 
                            float *O, float *l, float *m, 
                            int N, int d, float scale,
                            int Br, int Bc, int Tr, int Tc){

    // set index
    int tx = threadIdx.x;
    int bx = blockIdx.x; // Batch size
    int by = blockIdx.x; // number of heads

    // set global memory offset
    int qkv_offs = (bx * gridDim.y * N*d) + (by * N * d);
    int lm_offs = (bx * gridDim.y * N) + (by * N); // lm have no embedding, so no d

    int kv_tile = Bc * d;
    int qo_tile = Br * d;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = &smem[kv_tile];
    float* Vj = &smem[kv_tile * 2];
    float* Sij = &smem[kv_tile * 3];
    float* Oi = &smem[qo_tile + (kv_tile * 3)];

    // 5: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do
    for(int j = 0; j < Tc; j++){


        // 6: Load K𝑗, V𝑗 from HBM to on-chip SRAM
        for(int x = 0; x < d; x++){
            // tile * j -> which tile are we on
            // tx * d + x -> which element of the tile are we loading
            Kj[x + (tx * d)] = K[qkv_offs + (kv_tile * j) + (tx * d) + x]; // take one tile from gmem
            Vj[x + (tx * d)] = V[qkv_offs + (kv_tile * j) + (tx * d) + x];
        }
        __syncthreads();

        // 7: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do
        for(int i = 0; i < Tr; i++){

            for(int x = 0; x<d; x++){

                // 8: Load Q𝑖, O𝑖, ℓ𝑖, 𝑚𝑖 from HBM to on-chip SRAM
                Qi[x + (tx * d)] = V[qkv_offs + (qo_tile * j) + (tx * d) + x];
                Oi[x + (tx * d)] = O[qkv_offs + (qo_tile * j) + (tx * d) + x];



                // 9:  On chip, compute S𝑖𝑗 = Q𝑖 * K𝑗_transposed  ∈ R (𝐵𝑟×𝐵𝑐)
                // computing attention scores





            }

        }

    }


}


int main(){

    // static Br, Bc
    // test with example Q, K values to see if it properly loaded into smem
}