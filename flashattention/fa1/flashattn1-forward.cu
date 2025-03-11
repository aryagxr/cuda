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
    int bx = blockIdx.x;
    int by = blockIdx.x;

    // set global memory offset
    int qkv_offs = (bx * gridDim.y * N*d) + (by * N * d);
    int lm_offs = (bx * gridDim.y * N) + (by * N); // lm have no embedding, so no d

    int tile = Bc * d;
    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = &smem[tile];
    float* Vj = &smem[tile * 2];
    float* Sij = &smem[tile * 3];

    // 5: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do
    for(int j = 0; j < Tc; j++){


        // 6: Load K𝑗, V𝑗 from HBM to on-chip SRAM
        for(int x = 0; x < d; x++){
            // tile * j -> which tile are we on
            // tx * d + x -> which element of the tile are we loading
            Kj[x + (tx * d)] = K[qkv_offs + (tile * j) + (tx * d) + x]; // take one tile from gmem
            Vj[x + (tx * d)] = V[qkv_offs + (tile * j) + (tx * d) + x];

        }

    }


}


int main(){

    // static Br, Bc
}