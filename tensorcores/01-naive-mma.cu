#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define MATRIX_M 16
#define MATRIX_N 8
#define MATRIX_K 16

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
__global__ void tc_mma(half* A, half* B, float* C, float* D) {
    const int lane = threadIdx.x & 31; // lane id 0..31
    if (lane >= 32) return;

    __half a_h[8];
    __half b_h[4];
    float  c_f[4];

    //tile starting at A[0],B[0],C[0]
    half* A_tile = A; 
    half* B_tile = B;
    float* C_tile = C;

    int a_offset_base = (lane * 8) % (MATRIX_M * MATRIX_K); //16*16 = 256
    for (int i = 0; i < 8; ++i) {
        int idx = (a_offset_base + i) % (MATRIX_M * MATRIX_K);
        a_h[i] = A_tile[idx];
    }

    int b_offset_base = (lane * 4) % (MATRIX_K * MATRIX_N); //16*8 = 128
    for (int i = 0; i < 4; ++i) {
        int idx = (b_offset_base + i) % (MATRIX_K * MATRIX_N);
        b_h[i] = B_tile[idx];
    }

    int c_offset_base = (lane * 4) % (MATRIX_M * MATRIX_N); //16*8 = 128
    for (int i = 0; i < 4; ++i) {
        int idx = (c_offset_base + i) % (MATRIX_M * MATRIX_N);
        c_f[i] = C_tile[idx];
    }

    //pack bits to int
    const uint16_t *a_u16 = reinterpret_cast<const uint16_t *>(a_h);
    uint32_t A_packed[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t lo = (uint32_t)a_u16[2*i + 0];
        uint32_t hi = (uint32_t)a_u16[2*i + 1];
        A_packed[i] = (hi << 16) | lo;
    }

    const uint16_t *b_u16 = reinterpret_cast<const uint16_t *>(b_h);
    uint32_t B_packed[2];
    for (int i = 0; i < 2; ++i) {
        uint32_t lo = (uint32_t)b_u16[2*i + 0];
        uint32_t hi = (uint32_t)b_u16[2*i + 1];
        B_packed[i] = (hi << 16) | lo;
    }


    float Dfrag[4]; //output
    mma_m16n8k16(A_packed, B_packed, c_f, Dfrag);

    int out_base = lane * 4;
    for (int i = 0; i < 4; ++i) {
        D[out_base + i] = Dfrag[i];
    }

}

int main() {
    const int sizeA = MATRIX_M * MATRIX_K;
    const int sizeB = MATRIX_K * MATRIX_N;
    const int sizeC = MATRIX_M * MATRIX_N;

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
    float *dD; cudaMalloc(&dD, 32 * 4 * sizeof(float)); //outputs per lane

    cudaMemcpy(dA, hA, sizeA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel with single block of 32 threads (one warp)
    tc_mma<<<1, 32>>>(dA, dB, dC, dD);
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
