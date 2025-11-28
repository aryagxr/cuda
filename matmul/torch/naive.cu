#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>
#include <vector>



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

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {

    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);


    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    naive_matmul<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_cuda, "Naive matmul kernel");
}