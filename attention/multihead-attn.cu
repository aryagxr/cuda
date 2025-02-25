#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define WARP_SIZE 32

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    }



__global__ void compute_qkv(const float* inp, float* Q, float* K, float* V, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * NH * d;
    if (idx < total) {
        int b = (idx / (N * NH * d)) % B;
        int n = (idx / (NH * d)) % N;
        int h = (idx / d) % NH;
        int dim = idx % d;
        int C = NH * d;
        
        int q_offset = ((b * NH + h) * N + n) * d + dim;
        // int inp_offset = ((b * N + n) * 3 * NH * d) + (h * d) + dim;
        int inp_offset = (b * N * C) + ((n * 3 * NH * d) + (h * d) + dim);
        
        Q[q_offset] = inp[inp_offset];
        K[q_offset] = inp[inp_offset + NH * d];
        V[q_offset] = inp[inp_offset + 2 * NH * d];
    }
}


__global__ void scaled_dot_product_attention(const float* Q, const float* K, float* scores, int B, int NH, int N, int d, float scale) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    
    int q_offset = ((b * NH + h) * N + i) * d;
    int k_offset = (b * NH + h) * N * d;
    
    __shared__ float shared_K[WARP_SIZE][WARP_SIZE];
    float sum = 0.0f;
    
    for (int j = 0; j < N; j += WARP_SIZE) {
        if (j + threadIdx.y < N)
            shared_K[threadIdx.y][threadIdx.x] = K[k_offset + (j + threadIdx.y) * d + threadIdx.x];
        __syncthreads();
        
        for (int x = 0; x < WARP_SIZE; x++) {
            sum += Q[q_offset + x] * shared_K[x][threadIdx.y];
        }
        __syncthreads();
    }
    scores[((b * NH + h) * N + i) * N + threadIdx.y] = sum * scale;
}


__global__ void softmax(float* scores, int B, int NH, int N) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    
    int offset = ((b * NH + h) * N + i) * N;
    float max_val = -INFINITY;
    
    for (int j = 0; j < N; j++) {
        max_val = fmaxf(max_val, scores[offset + j]);
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < N; j++) {
        scores[offset + j] = expf(scores[offset + j] - max_val);
        sum_exp += scores[offset + j];
    }
    for (int j = 0; j < N; j++) {
        scores[offset + j] /= sum_exp;
    }
}



__global__ void compute_context_vector(const float* scores, const float* V, float* output, int B, int NH, int N, int d) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    
    int v_offset = ((b * NH + h) * N) * d;
    int s_offset = ((b * NH + h) * N + i) * N;
    int o_offset = ((b * NH + h) * N + i) * d;
    
    for (int x = 0; x < d; x++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += scores[s_offset + j] * V[v_offset + j * d + x];
        }
        output[o_offset + x] = sum;
    }
}


void multi_head_attention(float* out, const float* inp, int B, int N, int C, int NH) {
    int d = C / NH;
    float scale = 1.0f / sqrtf(d);
    
    float *Q, *K, *V, *scores;
    cudaMalloc(&Q, B * NH * N * d * sizeof(float));
    cudaMalloc(&K, B * NH * N * d * sizeof(float));
    cudaMalloc(&V, B * NH * N * d * sizeof(float));
    cudaMalloc(&scores, B * NH * N * N * sizeof(float));
    
    dim3 grid_qkv((B * N * NH * d + 255) / 256);
    compute_qkv<<<grid_qkv, 256>>>(inp, Q, K, V, B, N, NH, d);

    cudaDeviceSynchronize();
    
    
    dim3 grid_sdpa(B, NH);
    dim3 block_sdpa(N, N);
    scaled_dot_product_attention<<<grid_sdpa, block_sdpa>>>(Q, K, scores, B, NH, N, d, scale);
    cudaDeviceSynchronize();
    
    
    
    softmax<<<grid_sdpa, N>>>(scores, B, NH, N);
    cudaDeviceSynchronize();
    
    
    compute_context_vector<<<grid_sdpa, N>>>(scores, V, out, B, NH, N, d);
    cudaDeviceSynchronize();
    
    
    
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(scores);
}



void test_mha() {
    int B = 2;  // number of input sequence
    int N = 6;  
    int C = 4;  
    int NH = 2; 

    // int d = C / NH;
    int input_size = B * N * C;
    int output_size = B * N * C;

    float *h_inp = new float[input_size];
    float *h_out = new float[output_size];

    

    std::cout << "Input Embeddings:\n";
    for (int i = 0; i < input_size; i++) {
        h_inp[i] = static_cast<float>(rand()) / RAND_MAX; 
        std::cout << h_inp[i] << " ";
        if ((i + 1) % C == 0) { 
            std::cout << "\n";
        }
    }
    std::cout << std::endl;

    
    float *d_inp, *d_out;
    CHECK_CUDA(cudaMalloc(&d_inp, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, output_size * sizeof(float)));

   
    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, input_size * sizeof(float), cudaMemcpyHostToDevice));

    
    multi_head_attention(d_out, d_inp, B, N, C, NH);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost));



    std::cout << "Output:\n";
    for (int i = 0; i < B; i++) {
        std::cout << "Batch " << i << ":\n";
        for (int j = 0; j < N; j++) {
            std::cout << "[ ";
            for (int k = 0; k < C; k++) {
                std::cout << h_out[i * N * C + j * C + k] << " ";
            }
            std::cout << "]\n";
        }
    }


    delete[] h_inp;
    delete[] h_out;
    CHECK_CUDA(cudaFree(d_inp));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    test_mha();
    return 0;
}