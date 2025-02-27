#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>



#define CHECK_CUDA_CALL(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}





__global__ void fa1_forward(float *Q, float *K, float*V, float *O, float *l, float *m, int N, int Br, int Bc, int Tr, int Tc, int d, float scale){

    int tidx = threadIdx.x;

    // on chip smem for K,V
    extern __shared__ float smem[];
    float *Qi = smem;
    float *Kj = Qi + Bc * d;
    float *Vj = Kj + Bc * d;
    float *Oi = Vj + Bc * d;
    float *li = Oi + Br;
    float *mi = li + Br;




    //  5:  for 1 ≤ 𝑗 ≤ 𝑇𝑐 d
    for(int j = 1; j < Tc; j++){

        int k_offset = j * Bc * d;
        int v_offset = j * Bc * d;

        // 6: Load K𝑗, V𝑗 from HBM to on-chip SRA
        for (int i = tidx; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            Kj[row * d + col] = K[k_offset + i]; // Load K from global memory
            Vj[row * d + col] = V[v_offset + i]; // Load V from global memory
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
                Qi[row * d + col] = Q[Q_offset + t]; // Load Q from global memory
                Oi[row * d + col] = O[O_offset + t]; // Load O from global memory
            }

            for (int t = tidx; t < Br; t += blockDim.x) {
                li[t] = l[l_offset + t];
                mi[t] = m[m_offset + t];
            }
            __syncthreads();
        


        }



        

    }


}