#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <cuda_fp16.h>


#define BASE 10000.0f
#define SEQ_LEN 4
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

/*
    Kernel 1: 
    Using FP32, a naive implementation of RoPE.
    One thread processes one even/odd pair of input.
    Assuming single head, SEQ_LEN * d input matrix. 
*/


__global__ void kernel1_rope_naive(float* X, float* Ro, int d){

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;

    if(tidx < SEQ_LEN * d/2){
        int row = tidx / (d/2);
        int pair = tidx % (d/2);

        // global index of pair in gmem
        int g_idx = row * d + pair * 2;

        float x0 = X[g_idx];
        float x1 = X[g_idx + 1];

        // cos(m * theta), sin(m * theta)
        float theta = 1.0f / powf(BASE, (float)(pair/d));
        float m_theta = row * theta;
        float cos_theta = cosf(m_theta);
        float sin_theta = sinf(m_theta);

        // apply rotation to the pair
        Ro[g_idx] = x0 * cos_theta - x1 * sin_theta;
        Ro[g_idx + 1] = x1 * cos_theta + x0 * sin_theta;

    }
    
}


/*
    Kernel 2: 
    Using FP32 * 4, vectorized loads.
    One thread processes two even/odd pairs of input.
    One thread reads and writes 4 floats at a time.
    Assuming single head, SEQ_LEN * d input matrix. 
*/
__global__ void kernel2_rope_vectorized(float* X, float* Ro, int d){
    
    int tidx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
    if(tidx < SEQ_LEN * d/2){
        int row = tidx / (d/2);
        int pair = tidx % (d/2);

        int g_idx = row * d + pair * 2;

        float4 x = FLOAT4(X[g_idx]);
        float4 y;

        float theta = 1.0f / powf(BASE, (float)(pair/d));
        float m_theta = row * theta;
        float cos_theta = cosf(m_theta);
        float sin_theta = sinf(m_theta);

        float theta2 = 1.0f / powf(BASE, (float)((pair + 1) / d));
        float m_theta2 = row * theta2;
        float cos_theta2 = cosf(m_theta2);
        float sin_theta2 = sinf(m_theta2);

        y.x = x.x * cos_theta - x.y * sin_theta;
        y.y = x.y * cos_theta + x.x * sin_theta;
        y.z = x.z * cos_theta2 - x.w * sin_theta2;
        y.w = x.w * cos_theta2 + x.z * sin_theta2;

        FLOAT4(Ro[g_idx]) = y;


    }
    
}


/*
    Kernel 3: 
    Using FP16, half precision.
    One thread processes one even/odd pair of input.
    Assuming single head, SEQ_LEN * d input matrix. 
*/
__global__ void kernel3_rope_fp16(half* X, half* Ro, int d){

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;

    if(tidx < SEQ_LEN * d/2){
        int row = tidx / (d/2);
        int pair = tidx % (d/2);

        // global index of pair in gmem
        int g_idx = row * d + pair * 2;

        half x0 = X[g_idx];
        half x1 = X[g_idx + 1];

        // cos(m * theta), sin(m * theta)
        float theta_f = 1.0f / powf(BASE, (float)(pair / d));
        half theta = __float2half(theta_f);
        half m_theta = __hmul(__int2half_rn(row), theta);
        half cos_theta = hcos(m_theta);
        half sin_theta = hsin(m_theta);

        // apply rotation to the pair
        Ro[g_idx] = __hsub(__hmul(x0, cos_theta), __hmul(x1, sin_theta));
        Ro[g_idx + 1] = __hadd(__hmul(x1, cos_theta), __hmul(x0, sin_theta));

    }
    
}




int main(){
    const int d = 8; 
    const int seq_len = SEQ_LEN;
    const int num_elem = seq_len * d; 

 
    float h_X[num_elem];
    float h_Ro[num_elem];
    float h_Ro2[num_elem];
    half h_Ro3[num_elem];

    for (int i = 0; i < num_elem; i++) {
        h_X[i] = static_cast<float>(i);
    }

    std::cout << "Input Matrix X:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << h_X[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    float *d_X, *d_Ro, *d_Ro2;
    half *d_X_half, *d_Ro3;
    cudaMalloc(&d_X, num_elem * sizeof(float));
    cudaMalloc(&d_Ro, num_elem * sizeof(float));
    cudaMalloc(&d_Ro2, num_elem * sizeof(float));
    cudaMalloc(&d_X_half, num_elem * sizeof(half));
    cudaMalloc(&d_Ro3, num_elem * sizeof(half));


    cudaMemcpy(d_X, h_X, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_half, h_X, num_elem * sizeof(half), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (seq_len * d / 2 + blockSize - 1) / blockSize;

    // Time kernel1_rope_naive
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    kernel1_rope_naive<<<numBlocks, blockSize>>>(d_X, d_Ro, d);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float ms1 = 0.0f;
    cudaEventElapsedTime(&ms1, start1, stop1);

    cudaMemcpy(h_Ro, d_Ro, num_elem * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix Ro Naive Kernel:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << h_Ro[i * d + j] << " ";
        }
        std::cout << std::endl;
    }


    // Time kernel2_rope_vectorized
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    kernel2_rope_vectorized<<<numBlocks, blockSize>>>(d_X, d_Ro2, d);

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float ms2 = 0.0f;
    cudaEventElapsedTime(&ms2, start2, stop2);

    cudaMemcpy(h_Ro2, d_Ro2, num_elem * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix Ro from Vectorized Kernel:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << h_Ro2[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    // Time kernel3_rope_fp16
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);

    kernel3_rope_fp16<<<numBlocks, blockSize>>>(d_X_half, d_Ro3, d);

    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float ms3 = 0.0f;
    cudaEventElapsedTime(&ms3, start3, stop3);

    cudaMemcpy(h_Ro3, d_Ro3, num_elem * sizeof(half), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix Ro from FP16 Kernel:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << __half2float(h_Ro3[i * d + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Time taken by kernel1_rope_naive: " << ms1 << " ms" << std::endl;
    std::cout << "Time taken by kernel2_rope_vectorized: " << ms2 << " ms" << std::endl;
    std::cout << "Time taken by kernel3_rope_fp16: " << ms3 << " ms" << std::endl;

    cudaFree(d_X);
    cudaFree(d_Ro);
    cudaFree(d_Ro2);
    cudaFree(d_X_half);
    cudaFree(d_Ro3);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);


    return 0;

}




