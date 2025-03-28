#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

#define N 5

// deriv = 1 for x>=0, 0 for x<0
// dout -> upstream gradients (wrt loss)
// din -> output of backpass of relu
__global__ void relu_backpass(float* out, float* din, float* dout, int n) {
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (tidx < n) {
        float x = out[tidx];
        din[tidx] = (x >= 0) ? dout[tidx] : 0.0f;
    }
}

int main() {
    float h_in[N] = {-2.0, 0.0, 1.5, -3.0, 4.0};   // Forward input
    float h_dout[N] = {0.1, -0.2, 0.5, 0.3, -0.7}; // Incoming gradient
    float h_out[N] = {0.0, 0.0, 1.5, 0.0, 4.0};    // Sample forward pass output

    float *d_in, *d_out, *d_dout, *d_din;
    float h_din[N];


    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMalloc((void**)&d_dout, N * sizeof(float));
    cudaMalloc((void**)&d_din, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, N * sizeof(float), cudaMemcpyHostToDevice); // Use sample forward pass output
    cudaMemcpy(d_dout, h_dout, N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Forward Pass Output (ReLU):\n";
    for (int i = 0; i < N; i++) std::cout << h_out[i] << " ";
    std::cout << "\n";


    relu_backpass<<<1, N>>>(d_out, d_din, d_dout, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_din, d_din, N * sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << "Backward Pass Input (h_dout):\n";
    for (int i = 0; i < N; i++) std::cout << h_dout[i] << " ";
    std::cout << "\n";

    std::cout << "Backward Pass Output (Gradient h_din):\n";
    for (int i = 0; i < N; i++) std::cout << h_din[i] << " ";
    std::cout << "\n";


    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dout);
    cudaFree(d_din);

    return 0;
}