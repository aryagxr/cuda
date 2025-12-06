#include <stdio.h>
#include <stdlib.h>

#include <cuda_bf16.h>

#include <torch/extension.h>
#include <vector>



#define BLOCKSIZE 16
#define BM 64
#define BN 64
#define BK 8
#define NUM_C_PER_THD 8

__global__ void naive_matmul(float* A, float* B, float* C, int M, int K, int N){

    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < M && col < N){
        //each thread computes one element of the output C
        float acc = 0.0;
        for(int i=0; i<K; i++){
            acc += A[row * K + i] * B[i*N + col]; 
        }

        C[row * N + col] = acc;
        
    }
}


// coalesced + bf16
__global__ void naive_matmul_bfloat16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int K, int N){

    
    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < M && col < N){
        float acc = 0.0f;
        for(int i = 0; i < K; i++){
            acc += __bfloat162float(A[row * K + i]) * __bfloat162float(B[i * N + col]);
        }
        C[row * N + col] = __float2bfloat16(acc);
    }
}



//shared mem caching
__global__ void smem_tiled_matmul(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int K, int N){

    int trow = threadIdx.y;
    int tcol = threadIdx.x;

    //which tile/chunk are we currently in
    int row = blockIdx.y * BLOCKSIZE + trow;
    int col = blockIdx.x * BLOCKSIZE + tcol;

    float acc = 0.0f;

    __shared__ float sA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE][BLOCKSIZE];

    for(int bk = 0; bk < K; bk += BLOCKSIZE){

        //multiply a row of a with a col of b
        //move through a row in a, so keep track of col idx
        //move down a col in b, so keep track of row idx
        int a_col = bk + tcol;  
        int b_row = bk + trow;

        //load A tile into smem
        if(row < M && a_col < K){
            sA[trow][tcol] = __bfloat162float(A[row * K + a_col]);
        } else {
            sA[trow][tcol] = 0.0f;
        }

        //load b tile into smem
        if(b_row < K && col < N){
            sB[trow][tcol] = __bfloat162float(B[b_row * N + col]);
        } else {
            sB[trow][tcol] = 0.0f;
        }

        __syncthreads();

        //partial dotproduct
        //computing a blocksize * blocksize tile of C
        for(int i = 0; i < BLOCKSIZE; ++i){
            acc += sA[trow][i] * sB[i][tcol];
        }

        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = __float2bfloat16(acc);
    }
}



//1d blocktiling
//one warp computes one tile of output C

__global__ void blocktile_1d(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int K, int N){
    
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    //thread idx within block
    int tidy = threadIdx.y;
    int tidx = threadIdx.x;

    //row idx of output C matrix
    //each thread computes 4 cells in a row (so across columns)
    //col idx each thread
    int blockrow = blockIdx.y * BM;
    int blockcol = blockIdx.x * BN;

    int global_row = blockrow + (tidy * NUM_C_PER_THD);
    int global_col = blockcol + tidx;

    //linear index
    int tdx = tidy * BLOCKSIZE + tidx; 
    int numThreads = BLOCKSIZE * BLOCKSIZE; 


    //accumulate per thread output (4 outputs)
    float acc[NUM_C_PER_THD] = {0.f};

    //outerloop: loop through K tile by tile
    //innerloop: loop through each block, 1 thread 4 outputs
    for (int bk = 0; bk < K; bk += BK) {
        
        //load tile BM*BK
        // Each thread loads multiple elements to cover entire tile
        int numElementsA = BM * BK;
        for (int i = tdx; i < numElementsA; i += numThreads) {
            int localRow = i / BK;  //Which row in the tile (0 to BM-1)
            int localCol = i % BK;  //Which col in the tile (0 to BK-1)
            int globalRow = blockrow + localRow;
            int globalCol = bk + localCol;
            
            if (globalRow < M && globalCol < K) {
                sA[localRow * BK + localCol] = __bfloat162float(A[globalRow * K + globalCol]);
            } else {
                sA[localRow * BK + localCol] = 0.f;
            }
        }
        
        //load tile B BK*BN
        int numElementsB = BK * BN;
        for (int i = tdx; i < numElementsB; i += numThreads) {
            int localRow = i / BN;  //Which row in the tile (0 to BK-1)
            int localCol = i % BN;  //Which col in the tile (0 to BN-1)
            int globalRow = bk + localRow;
            int globalCol = blockcol + localCol;
            
            if (globalRow < K && globalCol < N) {
                sB[localRow * BN + localCol] = __bfloat162float(B[globalRow * N + globalCol]);
            } else {
                sB[localRow * BN + localCol] = 0.f;
            }
        }
        
        __syncthreads();
        
        
        for (int kk = 0; kk < BK; ++kk) {
            float bVal = sB[kk * BN + tidx];
            
            #pragma unroll
            for (int ii = 0; ii < NUM_C_PER_THD; ++ii) {
                int localRow = tidy * NUM_C_PER_THD + ii;
                float aVal = sA[localRow * BK + kk];
                acc[ii] += aVal * bVal;
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int ii = 0; ii < NUM_C_PER_THD; ++ii) {
        int r = global_row + ii;
        int c = global_col;
        if (r < M && c < N) {
            C[r * N + c] = __float2bfloat16(acc[ii]);
        }
    }
    

}





torch::Tensor matmul_naive_fp32(torch::Tensor A, torch::Tensor B) {
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE,
                (M + BLOCKSIZE - 1) / BLOCKSIZE);

    naive_matmul<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    return C;
}

torch::Tensor matmul_naive_bf16(torch::Tensor A, torch::Tensor B) {
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE,
                (M + BLOCKSIZE - 1) / BLOCKSIZE);

    naive_matmul_bfloat16<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, K, N
    );
    return C;
}

torch::Tensor matmul_smem_bf16(torch::Tensor A, torch::Tensor B) {
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE,
                (M + BLOCKSIZE - 1) / BLOCKSIZE);

    smem_tiled_matmul<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, K, N
    );
    return C;
}


torch::Tensor blocktiling_1d(torch::Tensor A, torch::Tensor B){
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    // dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE,
    //             (M + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 blocks((N + BN - 1) / BN,
            (M + BM - 1) / BM);

    blocktile_1d<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, K, N
    );
    return C;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive_fp32", &matmul_naive_fp32);
    m.def("matmul_naive_bf16", &matmul_naive_bf16);
    m.def("matmul_smem_bf16", &matmul_smem_bf16);
    m.def("blocktiling_1d", &blocktiling_1d);
}
