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
    int lm_offset = (bx * gridDim.y * N) + (by * N);

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
    float* lij = &smem[(qo_tile * 2) + (kv_tile * 2) + Si_tile];
    float* mij = &smem[(qo_tile * 2) + (kv_tile * 2) + Si_tile  + lm_tile];



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
            mij[tx] = -INFINITY;
            lij[tx] = 0.0f;
        }

        
        // Printing loaded mi and li values
        if (bx == 0 && by == 0 && tx == 0) { // Print for the first block only
            printf("Tile %d: mi values: ", i);
            for (int t = 0; t < Br; t++) {
                printf("%f ", mij[t]);
            }
            printf("\nTile %d: li values: ", i);
            for (int t = 0; t < Br; t++) {
                printf("%f ", lij[t]);
            }
            printf("\n");
        }
            

        // 6: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do
        for(int j = 0; j < Tc; j++){

            // 7: Load K𝑗, V𝑗 from HBM to on-chip SRAM
            for(int y = 0; y < d; y++){
                Kj[y + (tx * d)] = K[qkv_offset + (kv_tile * j) + (tx * d) + y];
                Vj[y + (tx * d)] = V[qkv_offset + (kv_tile * j) + (tx * d) + y];
            }


            
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
            

            // 8: On chip, compute Si j = Q𝑖K𝑇 ∈ (𝐵𝑟×𝐵𝑐)
            for(int c = 0; c < Bc; c++){
                float acc = 0.0f;
                for(int r = 0; r < d; r++){
                    acc += Qi[(tx * d) + r] * Kj[(c * d) + r];
                }
                Si[(tx * Bc) + c] = acc;
            }
            __syncthreads();

            // print Sij
            if (bx == 0 && by == 0 && tx == 0) {
                printf("Tile %d: Sij Computed:\n", j);
                for (int i = 0; i < Br; i++) {
                    for (int k = 0; k < Bc; k++) {
                        printf("%f ", Si[i * Bc + k]);
                    }
                    printf("\n");
                }
            }


            float mi_old = mij[tx]; // prev max
            if(tx < Br){

                float mi_new = -INFINITY;
                for(int y = 0; y < Bc; y++){

                    // max in that row
                    mi_new = max(mi_new, Si[tx * Bc + y]);
                }

                // 9: On chip, mi(j) = max( mi(j-1), rowmax(Si(j)) ) ∈ Br
                
                mij[tx] = max(mi_old, mi_new);

                // 9: Pij_dash = exp(sij - mij)
                float rowsum = 0.0f;
                for(int y=0; y<Bc; y++){
                    // just replacing Si matrix, saving smem space
                    Si[tx * Bc + y] = expf(Si[tx * Bc + y] - mij[tx]);
                    rowsum += Si[tx * Bc + y]; // rowsum of Pij_dash
                }
            
                // 9: ℓᵢ⁽ʲ⁾ = exp(mᵢ⁽ʲ⁻¹⁾ - mᵢ⁽ʲ⁾) * ℓᵢ⁽ʲ⁻¹⁾ + rowsum(P_ij_dash)
                lij[tx] = expf(mi_old - mij[tx]) * lij[tx] + rowsum;
            }
            __syncthreads();

            // print Pij
            if (bx == 0 && by == 0 && tx == 0) {
                printf("Tile %d: Pij Computed:\n", j);
                for (int i = 0; i < Br; i++) {
                    for (int k = 0; k < Bc; k++) {
                        printf("%f ", Si[i * Bc + k]);
                    }
                    printf("\n");
                }

                // Print mij values
                printf("Tile %d: mij values:\n", j);
                for (int i = 0; i < Br; i++) {
                    printf("%f ", mij[i]);
                }
                printf("\n");
                
                // Print lij values
                printf("Tile %d: lij values:\n", j);
                for (int i = 0; i < Br; i++) {
                    printf("%f ", lij[i]);
                }
                printf("\n");
            }
            


            // 10: On chip, compute Oi(j) = diag(e^(mi(j-1)-mi(j)))^(-1) * Oi(j-1) + P˜i(j) * Vj
            if(tx < Br){
                //float mscale = expf(mij[tx] - mi_old);
                float mscale = (mi_old == -INFINITY) ? 1.0f : expf(mij[tx] - mi_old);
                for(int x = 0; x < d; x++){
                    float pv_sum = 0.0f;

                    // Pij * Vj
                    for(int y = 0; y < Bc; y++){
                        pv_sum += Si[tx * Bc + y] * Vj[y * d + x];
                    }

                    // update output Oij
                    Oi[tx * d + x] =  mscale * Oi[tx * d + x] + pv_sum;
                }
            }
            __syncthreads();

            // Print Oi matrix after step 10
            if (bx == 0 && by == 0 && tx == 0) {
                printf("Tile %d: Oi matrix:\n", j);
                for (int i = 0; i < Br; i++) {
                    for (int x = 0; x < d; x++) {
                        printf("%f ", Oi[i * d + x]);
                    }
                    printf("\n");
                }
                printf("\n");
            }


        } // 11: end inner for

        // 12: Oi = diag(li (tc))^-1 Oi (Tc)
        if(tx < Br){
            for(int x = 0; x < d; x++){
                Oi[tx * d + x] = Oi[tx * d + x] / lij[tx];
            }


            // 13: On chip, compute Li = mi(Tc) + log(li(Tc))
            float Li = mij[tx] + logf(lij[tx]);

            // 15: Write Li to HBM as the i-th block of L
            l[lm_offset + (i * Br) + tx] = Li;

        }

        // 14: Write Oi to HBM as the i-th block of O
        for(int x = 0; x < d; x++){
            if(tx < Br) {
                O[qkv_offset + (qo_tile * i) + (tx * d) + x] = Oi[tx * d + x];
            }
        }

    } // outer for loop
    
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
    // for (int i = 0; i < BATCH * HEADS * SEQ_LEN; i++) {
    //     h_m[i] = -INFINITY;
    //     h_l[i] = 0;
    // }

    // for (int i = 0; i < total_size; i++) {
    //     h_O[i] = 0;
    // }


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

    int shared_mem_size = (2 * Br * EMBED_DIM) + (2 * Bc * EMBED_DIM) + (Br * Bc) + (2 * Br);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, shared_mem_size);

    /*--------------------------------------------------------------------*/
    
    cudaEventRecord(start);
    // Launch kernel
    fa2_forward<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                  SEQ_LEN, EMBED_DIM, 1.0f,
                                                  Br, Bc, Tr, Tc);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Flash-Attention 2 kernel execution time: %f ms\n", ms);

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