#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define BASE 10000.0f
#define SEQ_LEN 4

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
        float theta = 1.0f / powf(BASE, (float)(pair)/d);
        float m_theta = row * theta;
        float cos_theta = cosf(m_theta);
        float sin_theta = sinf(m_theta);

        // apply rotation to the pair
        Ro[g_idx] = x0 * cos_theta - x1 * sin_theta;
        Ro[g_idx + 1] = x1 * cos_theta + x0 * sin_theta;

    }
    
}





int main(){
    const int d = 8; 
    const int seq_len = SEQ_LEN;
    const int num_elem = seq_len * d; 

 
    float h_X[num_elem];
    float h_Ro[num_elem];

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

    float *d_X, *d_Ro;
    cudaMalloc(&d_X, num_elem * sizeof(float));
    cudaMalloc(&d_Ro, num_elem * sizeof(float));

    cudaMemcpy(d_X, h_X, num_elem * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (seq_len * d / 2 + blockSize - 1) / blockSize;
    kernel1_rope_naive<<<numBlocks, blockSize>>>(d_X, d_Ro, d);

    cudaMemcpy(h_Ro, d_Ro, num_elem * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix Ro:" << std::endl;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << h_Ro[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_X);
    cudaFree(d_Ro);
}




