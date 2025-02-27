#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void fa1_forward(float *Q, float *K, float*V, float *O, float *l, float *m, int N, int Br, int Bc, int Tr, int Tc, int d, float scale){

    int tidx = threadIdx.x;

    // on chip smem for K,V
    __shared__ float K_tile[Bc][d];
    __shared__ float V_tile[Bc][d];
    __shared__ float Q_tile[Br][d]; 
    __shared__ float O_tile[Br][d];
    __shared__ float l_tile[Br];
    __shared__ float m_tile[Br];


    int row_offset = blockIdx.x * Br;
    int col_offset = blockIdx.y * Bc;

    //  5:  for 1 ≤ 𝑗 ≤ 𝑇𝑐 d
    for(int j = 1; j < Tc; j++){

        int k_offset = j * Bc * d;
        int v_offset = j * Bc * d;

        // 6: Load K𝑗, V𝑗 from HBM to on-chip SRA
        for (int i = tidx; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            K_tile[row][col] = K[k_offset + i]; // Load from HBM (global memory)
            V_tile[row][col] = V[v_offset + i];
        }
        __syncthreads();


        // 7:  for 1 ≤ 𝑖 ≤ 𝑇𝑟 d
        for(int i = 1; i < Tr; i++){

            int Q_offset = i * Br * d;
            int O_offset = i * Br * d;
            int l_offset = i * Br;
            int m_offset = i * Br;

            // 8:  Load Q𝑖, O𝑖, ℓ𝑖, 𝑚𝑖 from HBM to on-chip SRAM
            for(int t = tidx; t < Br * d; t += blockDim.x){
                int row = t/d;
                int col = t%d;
                Q_tile[row][col] = Q[Q_offset + t];
                O_tile[row][col] = O[O_offset + t];
            }

            for (int t = tidx; t < Br; t += blockDim.x) {
                l_tile[t] = l[l_offset + t];
                m_tile[t] = m[m_offset + t];
            }
            __syncthreads();
        


        }



        

    }


}