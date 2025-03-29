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
#define sqrt2overPI 0.7978845608
#define k 0.044715


__global__ void gelu_backpass_fp32(float* in, float* grad_out, float* grad_in, int n) {
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tidx < n) {
        float x = in[tidx];
        float grad = grad_out[tidx];

        float tanh_arg = sqrt2overPI * (x + k * (x * x * x));
        float tanh_val = tanh(tanh_arg);
        float sech2 = 1 - tanh_val * tanh_val; // sech^2(x) = 1 - tanh^2(x)

        // deriv -> dGELU/dx
        float dgelu_dx = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * sqrt2overPI * (1.0f + 3.0f * k * (x * x));
        grad_in[tidx] = grad * dgelu_dx; //dL/dx
    }
}

int main() {
    // Host data
    float h_in[N] = {-2.0, 0.0, 1.5, -3.0, 4.0};   // Forward input
    float h_dout[N] = {0.1, -0.2, 0.5, 0.3, -0.7}; // Incoming gradient
    float h_din[N]; // Output gradient (to be computed)

    float *d_in, *d_dout, *d_din;


    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_dout, N * sizeof(float));
    cudaMalloc((void**)&d_din, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, h_dout, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gelu_backpass_fp32<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_dout, d_din, N);
    cudaDeviceSynchronize();


    cudaMemcpy(h_din, d_din, N * sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << "Backward Pass Input (h_dout):\n";
    for (int i = 0; i < N; i++) std::cout << h_dout[i] << " ";
    std::cout << "\n";

    std::cout << "Backward Pass Output (Gradient h_din):\n";
    for (int i = 0; i < N; i++) std::cout << h_din[i] << " ";
    std::cout << "\n";

    cudaFree(d_in);
    cudaFree(d_dout);
    cudaFree(d_din);

    return 0;
}