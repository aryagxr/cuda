
#include <iostream>
#include <cuda_runtime.h>

#define X 1024
#define Y 1024


// kernel
__global__ void MatMulNaive(float *A, float *B, float *C, int width, int height){

    // thread index
    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);


    if (col < width && row < height){
        float P = 0.0;
        // each thread computes one element of the output C matrix
        for (int i=0; i<width; i++){
            P += A[row* width +i] * B[i*width + col];
        }
        
        C[row * width + col] = P;

    }
    
}


int main(){

    // matrix size
    size_t matrix_size = X*Y*sizeof(float);

    // declare variables
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // malloc
    h_A = (float*)malloc(matrix_size);
    h_B = (float*)malloc(matrix_size);
    h_C = (float*)malloc(matrix_size);

    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    // matrix init
    for (int i = 0; i<X*Y; i++){
        h_A[i] = (i % X) + 1;   // 1, 2, 3, ..., N, columnwise
        h_B[i] = (i / Y) + 1;  // 1,2,3...N rowwise
    }


    // transfer from host to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    // grid dimension
    dim3 threadsPerBlock(16,16); //256 threads in each block
    dim3 blocksPerGrid((X + threadsPerBlock.x - 1) / threadsPerBlock.x,  // 1024*1024 = 1048576 threads needed
                        (Y + threadsPerBlock.y - 1) / threadsPerBlock.y); // 1048576 / 256 = 4096 blocks needed

    
    MatMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, X, Y);
    cudaDeviceSynchronize();

    // transfer back to host
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);

    printf("Successful");
    printf("Sample result C[0][0] = %f\n", h_C[0]);


    // free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;

}