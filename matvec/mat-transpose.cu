// matrix transpose

#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define M 1024

__global__ void matTranspose(float *A, float *A_T, int m, int n){

    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < m && col < n){
        A_T[col*m + row] = A[row*n + col]; //swap row and col
    }
}


int main(){

    size_t mat_size = M*N*sizeof(float);
    float *hA, *hA_t;
    float *dA, *dA_t;

    hA = (float*)malloc(mat_size);
    hA_t = (float*)malloc(mat_size);

    for(int i=0; i<M*N; i++){
        hA[i]=i+1;
    }

    cudaMalloc((void**)&dA, mat_size);
    cudaMalloc((void**)&dA_t, mat_size);

    cudaMemcpy(dA, hA, mat_size, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(32,32); // 1024 threads per block
    dim3 blocksPerGrid(32,32); //4 blocks -> 4*256=1024 threads

    matTranspose<<<blocksPerGrid, threadsPerBlock>>>(dA, dA_t, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hA_t, dA_t, mat_size, cudaMemcpyDeviceToHost);

    
    
    // for (int j = 0; j < N; j++) {
    //     for (int i = 0; i < M; i++) {
    //         printf("%f ", hA_t[j * M + i]);
    //     }
    //     printf("\n");
    // }
    printf("Successful");

    cudaFree(dA); cudaFree(dA_t);
    free(hA); free(hA_t);
}

