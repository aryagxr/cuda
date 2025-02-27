#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)




__global__ void fa1_forward(float *Q, float *K, float*V, float *O, float *l, float *m, int N, int Br, int Bc, int Tr, int Tc, int d, float scale){

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;  // Block index (useful for debugging)

    // on chip smem for K,V
    extern __shared__ float smem[];
    float *Qi = smem;
    float *Kj = Qi + Bc * d;
    float *Vj = Kj + Bc * d;
    float *Oi = Vj + Bc * d;
    float *li = Oi + Br;
    float *mi = li + Br;
    float *Sij = mi + Br;
    float *mij_dash = Sij + Br;
    float *mi_new = mij_dash + Br;
    float *li_new = mi_new + Br;
    float *Pij = li_new + Br;
    float *li_dash = Pij + Br * Bc;




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




            // 9: Compute S𝑖𝑗 = Q𝑖 * K𝑇𝑗 ∈ R𝐵𝑟×𝐵c
            for(int r = 0; r < Br; r++){
                float sum_acc = 0.0f;
                for(int c = 0; c < d; c++){
                    sum_acc += Qi[r * d + c] * Kj[j * Bc * d + c];
                }
                Sij[r * Bc + j] = sum_acc;
            }
            __syncthreads();


            // 10: On chip, compute 𝑚𝑖𝑗 = rowmax(S𝑖𝑗) ∈ R𝐵𝑟, P𝑖𝑗 = exp(S𝑖𝑗 − 𝑚𝑖𝑗) ∈ R𝐵𝑟×𝐵𝑐 (pointwise), ℓ𝑖𝑗 =rowsum(P𝑖𝑗) ∈ R𝐵𝑟
            // compute row max of a block  of Sij
            for(int r = tidx; r < Br; r+=blockDim.x){
                float max_val = -INFINITY;
                for(int c = 0; c < Bc; c++){
                    max_val = fmaxf(max_val, Sij[r * Bc + c]);
                }
                mij_dash[r] = max_val;
            }
            __syncthreads();


            // compute Pij exp(Sij - mij_dash) - numerator of softmax & li_new - denominator of softmax
            for(int r = tidx; r < Br; r+=blockDim.x){
                float row_l = 0.0f;
                for(int c = 0; c < Bc; c++){
                    Pij[r * Bc + c] = expf(Sij[r * Bc + c] - mij_dash[r]);
                    row_l += Pij[r * Bc + c];
                }
                li_dash[r] = row_l;
                
            }
            __syncthreads();


            // 11: On chip, compute 𝑚𝑖_new = max(mij_dash, mi) and li_new
            for(int r = tidx; r < Br; r+=blockDim.x){
                mi_new[r] = fmaxf(mij_dash[r], mi[r]);
                li_new[r] = expf(mi[r] - mi_new[r]) * li[r] + expf(mij_dash[r] - mi_new[r]) * li_dash[r];
            }
            __syncthreads();


            // 12: Write O𝑖 ← diag(ℓi_new)^-1 (diag(ℓ𝑖)𝑒^𝑚𝑖−𝑚new𝑖 O𝑖 + 𝑒𝑚˜𝑖 𝑗−𝑚new𝑖 P𝑖𝑗 V𝑗) to HBM
            for(int r = tidx; r < Br; r += blockDim.x){
                float scale_inv = 1.0f / li_new[r];

                for(int c = 0; c < d; c++){
                    float old_Oi = li[r] * expf(mi[r] - mi_new[r]) * Oi[r * d + c];
                    
                    float new_Oi = 0.0f;
                    for(int k = 0; k < Bc; k++){
                        new_Oi += Pij[r * Bc + k] * Vj[k * d + c];
                    }
                    new_Oi *= expf(mij_dash[r] - mi_new[r]);

                    Oi[r * d + c] = scale_inv * (old_Oi + new_Oi);
                }
            }
            __syncthreads();

            for(int t = tidx; t < Br * d; t += blockDim.x){
                O[O_offset + t] = Oi[t];
            }

            for(int r = tidx; r < Br; r += blockDim.x){
                l[l_offset + r] = li_new[r];  // Store updated ℓ𝑖
                m[m_offset + r] = mi_new[r];  // Store updated m𝑖
            }
            __syncthreads();
        }
    }
}

            




void printMatrix(float *matrix, int rows, int cols, const char *name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {


    const int N = 8;  // Batch size
    const int Br = 8;  // Row block size
    const int Bc = 8;  // Column block size
    const int Tr = 1;  // Row tile size
    const int Tc = 1;  // Column tile size
    const int d = 8;  // Hidden size
    const int BLOCK_SIZE = 256;

    int size = N * d * sizeof(float);
    int vec_size = N * sizeof(float);

    // Allocate host memory
    float h_Q[N * d] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64
    };

    float h_K[N * d] = {
        1, 0, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 0, 0, 1,
        1, 1, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 1, 1, 0,
        1, 0, 1, 1, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 0, 0, 1
    };

    float h_V[N * d] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64
    };

    float h_O[N * d] = {0};  // Output matrix initialized to 0
    float h_l[N] = {0};  // Scaling factors
    float h_m[N] = {0};

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc((void**)&d_Q, size);
    cudaMalloc((void**)&d_K, size);
    cudaMalloc((void**)&d_V, size);
    cudaMalloc((void**)&d_O, size);
    cudaMalloc((void**)&d_l, vec_size);
    cudaMalloc((void**)&d_m, vec_size);

    // Copy data to device
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, vec_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int shared_mem_size = (Br * d + Bc * d + Bc * d + Br + Br + Br + Br + Br + Br + Br * Bc + Br) * sizeof(float);
    fa1_forward<<<1, BLOCK_SIZE, shared_mem_size>>>(d_Q, d_K, d_V, d_O, d_l, d_m, N, Br, Bc, Tr, Tc, d, 1.0f);

    // Copy results back to host
    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

    // Print matrices
    printMatrix(h_Q, N, d, "Q");
    printMatrix(h_K, N, d, "K");
    printMatrix(h_V, N, d, "V");
    printMatrix(h_O, N, d, "O (Output)");

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);

    return 0;
}
