#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


//TODO: Reduce if(tx < Br) checks 


__global__ void fa1_forward(float *Q, float *K, float*V, 
                            float *O, float *l, float *m, 
                            int N, int d, float scale,
                            int Br, int Bc, int Tr, int Tc){

    // set index
    int tx = threadIdx.x;
    int bx = blockIdx.x; // Batch size
    int by = blockIdx.y; // number of heads

    // set global memory offset
    int qkv_offs = (bx * gridDim.y * N*d) + (by * N * d);
    int lm_offs = (bx * gridDim.y * N) + (by * N); // lm have no embedding, so no d

    int kv_tile = Bc * d;
    int qo_tile = Br * d;
    int Sij_tile = Br * Bc;
    int lm_tile = Br;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = &smem[kv_tile];
    float* Vj = &smem[kv_tile * 2];
    float* Sij = &smem[kv_tile * 3];
    float* Oi = &smem[Sij_tile + (kv_tile * 3)];
    float* mi = &smem[Sij_tile + (kv_tile * 3) + qo_tile];
    float* li = &smem[Sij_tile + (kv_tile * 3) + qo_tile + lm_tile];
    float* m_ij = &smem[Sij_tile + (kv_tile * 3) + qo_tile + (2 * lm_tile)];
    float* l_ij = &smem[Sij_tile + (kv_tile * 3) + qo_tile + (3 * lm_tile)];
    float* mi_new = &smem[Sij_tile + (kv_tile * 3) + qo_tile + (4 * lm_tile)];
    float* li_new = &smem[Sij_tile + (kv_tile * 3) + qo_tile + (5 * lm_tile)];


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


        /*
        if (bx == 0 && by == 0 && tx == 0) { // Ensure only one block prints
            printf("Tile %d: Shared K: ", j);
            for (int i = 0; i < kv_tile; i++) {
                printf("%f ", Kj[i]);
            }
            printf("\nTile %d: Shared V: ", j);
            for (int i = 0; i < kv_tile; i++) {
                printf("%f ", Vj[i]);
            }
            printf("\n");
        }
        */
        

        // 7: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do - inner loop
        for(int i = 0; i < Tr; i++){

            for(int x = 0; x<d; x++){

                // 8: Load Q𝑖, O𝑖, ℓ𝑖, 𝑚𝑖 from HBM to on-chip SRAM
                Qi[x + (tx * d)] = Q[qkv_offs + (qo_tile * i) + (tx * d) + x];
                Oi[x + (tx * d)] = O[qkv_offs + (qo_tile * i) + (tx * d) + x];

            }
            if (tx < Br) {
                mi[tx] = m[lm_offs + (i * Br) + tx];  // Load mi from gmem
                li[tx] = l[lm_offs + (i * Br) + tx];  // Load li from gmem
            }
            __syncthreads();


            /*
            // Printing loaded mi and li values
            if (bx == 0 && by == 0 && tx == 0) { // Print for the first block only
                printf("Tile %d: mi values: ", j);
                for (int t = 0; t < Br; t++) {
                    printf("%f ", mi[t]);
                }
                printf("\nTile %d: li values: ", j);
                for (int t = 0; t < Br; t++) {
                    printf("%f ", li[t]);
                }
                printf("\n");
            }
            */
            
            

            // 9:  On chip, compute S𝑖𝑗 = Q𝑖 * K𝑗_transposed  ∈ R (𝐵𝑟×𝐵𝑐)
            // computing attention scores Sij
            for(int r = 0; r < d; r++){
                for(int c = 0; c < Bc; c++){
                    Sij[(tx * Bc) + c] += Qi[tx*d + r] * Kj[(c * d) + r];
                }
            }

            
            /*
            // print Sij
            if (bx == 0 && by == 0 && tx == 0) {
                printf("Tile %d: Sij Computed:\n", j);
                for (int i = 0; i < Br; i++) {
                    for (int k = 0; k < Bc; k++) {
                        printf("%f ", Sij[i * Bc + k]);
                    }
                    printf("\n");
                }
            }
            */
            

            
            if (tx < Br) {
                m_ij[tx] = -INFINITY;
                l_ij[tx] = 0;
            }
            __syncthreads();

            // 10:  On chip, compute 𝑚˜𝑖𝑗 = rowmax(Sij)
            if (tx < Br) {
                for (int y = 0; y < Bc; y++) {
                    m_ij[tx] = max(m_ij[tx], Sij[tx * Bc + y]);
                }

                // 11: On chip compute mi_new = max(mi, m_ij)
                mi_new[tx] = max(mi[tx], m_ij[tx]);
            }
            __syncthreads();



            /*
            // print m_ij rowmax values
            if (bx == 0 && by == 0 && tx == 0) {
                printf("Tile %d: m_ij values:\n", j);
                for (int i = 0; i < Br; i++) {
                    printf("%f ", m_ij[i]);
                }
                printf("\n");


                printf("Tile %d: mi_new values:\n", j);
                for (int i = 0; i < Br; i++) {
                    printf("%f ", mi_new[i]);
                }
                printf("\n");
            }
            */    


            
            // one thread per row
            if(tx < Br){

                // 10:  P˜𝑖𝑗 = exp(S𝑖𝑗 − 𝑚˜𝑖𝑗) & l_ij = rowsum(P_ij)
                // instead of creating a Pij matrix, edit Sij in place
                for(int x=0; x < Bc; x++){
                    Sij[(tx * Bc) + x] = expf(Sij[(tx * Bc) + x] - m_ij[tx]);
                    l_ij[tx] += Sij[(tx * Bc) + x];  
                }

                // 11: on chip, li_new = e^(mi-mi_new)li + e^(m_ij-mi_new)l_ij
                li_new[tx] = expf(mi[tx] - mi_new[tx]) + expf(m_ij[tx] - mi_new[tx]);

            }
            __syncthreads();


            /*
            // print Sij, which now has Pij values and l_ij
            if (bx == 0 && by == 0 && tx == 0) { 
                printf("Sij matrix (Br x Bc):\n");
                for (int i = 0; i < Br; i++) {
                    for (int k = 0; k < Bc; k++) {
                        printf("%f ", Sij[i * Bc + k]);
                    }
                    printf("\n");
                }

                printf("L_ij values for tile %d:\n", j);
                for (int t = 0; t < Br; t++) {
                    printf("L_ij[%d] = %f\n", t, l_ij[t]);
                }

                printf("Li_new values for tile %d:\n", j);
                for (int t = 0; t < Br; t++) {
                    printf("Li_new[%d] = %f\n", t, li_new[t]);
                }
            }
            */


            // 12: Update Oi and write it back to HBM
            // Oi ← diag(li_new)^(-1) * (diag(li) * e^(mi - mi_new) * Oi + e^(m_ij - mi_new) * P_ij * Vj)
            if(tx < Br){

                float pv = 0.0f;

                for(int y = 0; y < Bc; y++) {
                    pv += Sij[tx * Bc + y] * Vj[y * d + tx];  
                }

                for(int x = 0; x < d; x++) {
                    Oi[tx * d + x] = (1 / li_new[tx]) * (
                        (li[tx] * expf(mi[tx] - mi_new[tx]) * Oi[tx * d + x])  // diag(li) * e^(mi - mi_new) * Oi
                        + (expf(m_ij[tx] - mi_new[tx]) * pv)  // e^(m_ij - mi_new) * P_ij * Vj
                    );
                    O[qkv_offs + (qo_tile * j) + (tx * d) + x] = Oi[tx * d + x];
                }

                

                // 13: Write li, mi to HBM
                m[lm_offs + (j * Br) + tx] = mi_new[tx];
                l[lm_offs + (j * Br) + tx] = li_new[tx];
            }
            __syncthreads();


            /*
            if (bx == 0 && by == 0 && tx == 0) {
                printf("Final Oi values:\n");
                for (int i = 0; i < Br; i++) {
                    printf("Oi[%d]: ", i);
                    for (int x = 0; x < d; x++) {
                        printf("%f ", Oi[i * d + x]);
                    }
                    printf("\n");
                }
            }
            */

        } // 14: close innerloop

    } // 15: close outerloop

}




int main() {

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
    float *h_m = new float[BATCH * HEADS * SEQ_LEN];  // New array for m
    float *h_l = new float[BATCH * HEADS * SEQ_LEN];  // New array for l
    float *h_O = new float[total_size];

    // Initialize Q, K, V with known values
    for (int b = 0; b < BATCH; b++) {
        for (int h = 0; h < HEADS; h++) {
            for (int s = 0; s < SEQ_LEN; s++) {
                for (int e = 0; e < EMBED_DIM; e++) {
                    int idx = ((b * HEADS + h) * SEQ_LEN + s) * EMBED_DIM + e;
                    h_Q[idx] = idx + 1;  // Sequential values for easy tracking
                    h_K[idx] = idx + 10;
                    h_V[idx] = idx + 20;
                }
            }
        }
    }


    // Initialize m with -infinity and l with 0
    for (int i = 0; i < BATCH * HEADS * SEQ_LEN; i++) {
        h_m[i] = -INFINITY;
        h_l[i] = 0;
    }

    for (int i = 0; i < total_size; i++) {
        h_O[i] = 0;
    }


        // Print input K matrix
    std::cout << "Input K Matrix:\n";
    for (int b = 0; b < BATCH; b++) {
        for (int h = 0; h < HEADS; h++) {
            std::cout << "Batch " << b << ", Head " << h << ":\n";
            for (int s = 0; s < SEQ_LEN; s++) {
                for (int e = 0; e < EMBED_DIM; e++) {
                    int idx = ((b * HEADS + h) * SEQ_LEN + s) * EMBED_DIM + e;
                    std::cout << h_K[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // Print input V matrix
    std::cout << "Input V Matrix:\n";
    for (int b = 0; b < BATCH; b++) {
        for (int h = 0; h < HEADS; h++) {
            std::cout << "Batch " << b << ", Head " << h << ":\n";
            for (int s = 0; s < SEQ_LEN; s++) {
                for (int e = 0; e < EMBED_DIM; e++) {
                    int idx = ((b * HEADS + h) * SEQ_LEN + s) * EMBED_DIM + e;
                    std::cout << h_V[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // Print the output matrix O
    std::cout << "Output O Matrix:\n";
    for (int b = 0; b < BATCH; b++) {
        for (int h = 0; h < HEADS; h++) {
            std::cout << "Batch " << b << ", Head " << h << ":\n";
            for (int s = 0; s < SEQ_LEN; s++) {
                for (int e = 0; e < EMBED_DIM; e++) {
                    int idx = ((b * HEADS + h) * SEQ_LEN + s) * EMBED_DIM + e;
                    std::cout << h_O[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    /*--------------------------------------------------------------------*/

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc(&d_Q, total_size * sizeof(float));
    cudaMalloc(&d_K, total_size * sizeof(float));
    cudaMalloc(&d_V, total_size * sizeof(float));
    cudaMalloc(&d_O, total_size * sizeof(float));
    cudaMalloc(&d_l, BATCH * HEADS * SEQ_LEN * sizeof(float));
    cudaMalloc(&d_m, BATCH * HEADS * SEQ_LEN * sizeof(float));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    /*--------------------------------------------------------------------*/

    cudaEventRecord(start);

    // Copy data to device
    cudaMemcpy(d_Q, h_Q, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, BATCH * HEADS * SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, BATCH * HEADS * SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h_O, d_O, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    /*--------------------------------------------------------------------*/

    // Define grid and block size
    dim3 grid(BATCH, HEADS);
    dim3 block(SEQ_LEN);

    int shared_mem_size = (Br * EMBED_DIM) + (2 * Bc * EMBED_DIM) + (Br * Bc) + (Br * EMBED_DIM);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, shared_mem_size);

    /*--------------------------------------------------------------------*/
    
    cudaEventRecord(start);
    // Launch kernel
    fa1_forward<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                  SEQ_LEN, EMBED_DIM, 1.0f,
                                                  Br, Bc, Tr, Tc);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Flash-Attention 1 kernel execution time: %f ms\n", ms);

    /*--------------------------------------------------------------------*/

    cudaEventRecord(start);
    cudaMemcpy(h_O, d_O, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    /*--------------------------------------------------------------------*/

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print the final output O matrix
    std::cout << "Final Output O Matrix:\n";
    for (int b = 0; b < BATCH; b++) {
        for (int h = 0; h < HEADS; h++) {
            std::cout << "Batch " << b << ", Head " << h << ":\n";
            for (int s = 0; s < SEQ_LEN; s++) {
                for (int e = 0; e < EMBED_DIM; e++) {
                    int idx = ((b * HEADS + h) * SEQ_LEN + s) * EMBED_DIM + e;
                    std::cout << h_O[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // Free memory
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_m;
    delete[] h_l;
    delete[] h_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);

    return 0;
}