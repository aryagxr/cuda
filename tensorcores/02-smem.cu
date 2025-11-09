#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// #define MATRIX_M 16
// #define MATRIX_N 8
// #define MATRIX_K 16



static inline __device__ void mma_m16n8k16(const uint32_t *Afrag, const uint32_t *Bfrag, const float *Cfrag, float *Dfrag) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(Dfrag[0]), "=f"(Dfrag[1]), "=f"(Dfrag[2]), "=f"(Dfrag[3])
        : "r"(Afrag[0]), "r"(Afrag[1]), "r"(Afrag[2]), "r"(Afrag[3]),
          "r"(Bfrag[0]), "r"(Bfrag[1]),
          "f"(Cfrag[0]), "f"(Cfrag[1]), "f"(Cfrag[2]), "f"(Cfrag[3])
    );
}

// each warp does one m16n8k16 tile.
// launch with <<<1,32>>> (one block of one warp)
__global__ void tc_mma(half* A, half* B, float* C, float* D, int M, int N, int K) {

    int tx = threadIdx.x;//0,1,..15
    int ty = threadIdx.y; //0,1,..15
    int tidx = ty * 16 + tx;

    int warp = tidx >> 5; //=tidx/(2^5)
    int lane = tidx & 31; // lane id 0..31

    //warp grid with 2 rows 4 cols
    int warp_row = warp / 4;
    int warp_col = warp % 4;

    //which row and col (16*8 tile) of the output
    int row_warp_offset = warp_row * 16;
    int col_warp_offset = warp_col * 8;

    // each block computes 32*32 tile of output D
    int blockM = blockIdx.y * 32;
    int blockN = blockIdx.x * 32;

    //8 warps compute 32*32 tile of output D
    //so A(32*16) * B(16*32) = D(32*32)
    __shared__ half A_shared[32][16];
    __shared__ half B_shared[16][32];

    int mWarp = row_warp_offset; // 0 or 16
    int nWarp = col_warp_offset; // 0,8,16,24

    // compute groupID and tid_in_group for lane in warp
    int groupID = lane >> 2; // 0 to 7
    int tid_in_group = lane & 3; // 0 to 3
    float dReg[4];
    
    for (int i = 0; i < 4; ++i) {
        int local_row = (i < 2) ? groupID : groupID + 8;
        int local_col = (tid_in_group * 2) + (i & 1);
        int gm = blockM + row_warp_offset + local_row;
        int gn = blockN + col_warp_offset + local_col;
        if (gm < M && gn < N) dReg[i] = C[gm * N + gn];
        else dReg[i] = 0.0f;
    }


    for (int k_start = 0; k_start < K; k_start += 16){
        int local_row = ty;
        int local_col = tx;

        //load A tile to smem
        for (int m = 0; m < 2; ++m) {
            int sm_row = m*16 + local_row;// 0 to 31
            int g_row = blockM + sm_row; // global row
            int g_k   = k_start + local_col; // global k index (0 to K)
            half val = (g_row < M && g_k < K) ? A[g_row * K + g_k] : __float2half(0.0f);
            A_shared[sm_row][local_col] = val;
        }

        // load B tile to smem
        for (int n = 0; n < 2; ++n) {
            int sm_col = n*16 + local_col;// 0 to 31
            int g_col = blockN + sm_col; // global col
            int g_k   = k_start + local_row; // global k index
            half val = (g_k < K && g_col < N) ? B[g_k * N + g_col] : __float2half(0.0f);
            B_shared[local_row][sm_col] = val; 
        }
        __syncthreads();

        

        __half a_h[8];
        __half b_h[4];
        //float  c_f[4];

        //loading fragments (fragment layout frm docs)
        a_h[0] = A_shared[mWarp + groupID    ][(tid_in_group*2)    ];
        a_h[1] = A_shared[mWarp + groupID    ][(tid_in_group*2) + 1];
        a_h[2] = A_shared[mWarp + groupID + 8][(tid_in_group*2)    ];
        a_h[3] = A_shared[mWarp + groupID + 8][(tid_in_group*2) + 1];
        a_h[4] = A_shared[mWarp + groupID    ][(tid_in_group*2) + 8];
        a_h[5] = A_shared[mWarp + groupID    ][(tid_in_group*2) + 9];
        a_h[6] = A_shared[mWarp + groupID + 8][(tid_in_group*2) + 8];
        a_h[7] = A_shared[mWarp + groupID + 8][(tid_in_group*2) + 9];


        // load B fragment (4 halfs) from Bs_shared
        b_h[0] = B_shared[(tid_in_group*2) + 0][ nWarp + groupID ];
        b_h[1] = B_shared[(tid_in_group*2) + 1][ nWarp + groupID ];
        b_h[2] = B_shared[(tid_in_group*2) + 8][ nWarp + groupID ];
        b_h[3] = B_shared[(tid_in_group*2) + 9][ nWarp + groupID ];

        // pack A into u16
        const uint16_t *a_u16 = reinterpret_cast<const uint16_t *>(a_h);
        uint32_t A_packed[4];
        for (int i = 0; i < 4; ++i) {
            uint32_t lo = (uint32_t)a_u16[2*i + 0];
            uint32_t hi = (uint32_t)a_u16[2*i + 1];
            A_packed[i] = (hi << 16) | lo;
        }
        // pack B into u16
        const uint16_t *b_u16 = reinterpret_cast<const uint16_t *>(b_h);
        uint32_t B_packed[2];
        for (int i = 0; i < 2; ++i) {
            uint32_t lo = (uint32_t)b_u16[2*i + 0];
            uint32_t hi = (uint32_t)b_u16[2*i + 1];
            B_packed[i] = (hi << 16) | lo;
        }

        
        float tmpOut[4];
        mma_m16n8k16(A_packed, B_packed, dReg, tmpOut);
        for (int i = 0; i < 4; ++i) dReg[i] = tmpOut[i];

    }


    // write back to global D
    for (int i = 0; i < 4; ++i) {
        int local_row = (i < 2) ? groupID : groupID + 8; 
        int local_col = (tid_in_group * 2) + (i & 1); // local column in 16x8 tile
        int gm = blockM + row_warp_offset + local_row;
        int gn = blockN + col_warp_offset + local_col;
        if (gm < M && gn < N) {
            D[gm * N + gn] = dReg[i];
        }
}



}

int main() {
    int M = 128;
    int N = 128;
    int K = 128;
    const int sizeA = M * K;
    const int sizeB = K * N;
    const int sizeC = M * N;

    half *hA = (half*)malloc(sizeA * sizeof(half));
    half *hB = (half*)malloc(sizeB * sizeof(half));
    float *hC = (float*)malloc(sizeC * sizeof(float));
    float *hD = (float*)malloc(32 * 4 * sizeof(float));

    for (int i = 0; i < sizeA; ++i) hA[i] = __float2half( (float)(i % 7 + 1) * 0.1f );
    for (int i = 0; i < sizeB; ++i) hB[i] = __float2half( (float)(i % 5 + 1) * 0.2f );
    for (int i = 0; i < sizeC; ++i) hC[i] = 0.0f; 

    half  *dA; cudaMalloc(&dA, sizeA * sizeof(half));
    half  *dB; cudaMalloc(&dB, sizeB * sizeof(half));
    float *dC; cudaMalloc(&dC, sizeC * sizeof(float));
    float *dD; cudaMalloc(&dD, sizeC * sizeof(float)); //outputs per lane

    cudaMemcpy(dA, hA, sizeA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    
    dim3 blockDim(16, 16); //256 threads/32 = 8 warps
    //each thread block(8 warps together) computes a 32*32 tile of the output D
    dim3 gridDim((N+31)/32,(M+31)/32); 
    tc_mma<<<gridDim, blockDim>>>(dA, dB, dC, dD, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(hD, dD, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results from first 8 lanes
    printf("Results (per-lane D fragments) — first 8 lanes:\n");
    for (int lane = 0; lane < 8; ++lane) {
        printf("lane %2d: ", lane);
        for (int j = 0; j < 4; ++j) {
            printf("%10.6f ", hD[lane*4 + j]);
        }
        printf("\n");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    free(hA); free(hB); free(hC); free(hD);
    return 0;
}
