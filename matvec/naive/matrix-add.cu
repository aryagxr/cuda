// adding matrices

#include <iostream>
#include <cuda_runtime.h>

// define kernel
__global__ void MatAdd(const int *A, int *B, int*C, int x_axis, int y_axis){

    // thread index 2D grid
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    // global index
    int idx = (row * y_axis) + col;

    if(col < x_axis && row < y_axis){
        C[idx] = A[idx] + B[idx];
    }
    
}


int main(){

    // initialize number of bytes to allocate (4*4 matrix)
    // memory needed
    const int N = 4;
    const int matrix_size = N*N*sizeof(int);

    // initialize the matrix elements
    int h_A[N][N] = {{1, 2, 3, 4}, 
                     {5, 6, 7, 8}, 
                     {9, 10, 11, 12}, 
                     {13, 14, 15, 16}};  // Input matrix A
    int h_B[N][N] = {{16, 15, 14, 13}, 
                     {12, 11, 10, 9}, 
                     {8, 7, 6, 5}, 
                     {4, 3, 2, 1}};  // Input matrix B
    int h_C[N][N];  // Output matrix C



    // initialize device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrix_size);
    cudaMalloc((void **)&d_B, matrix_size);
    cudaMalloc((void **)&d_C, matrix_size);



    // transfer variable from host to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);


    // block and grid dim
    dim3 dimGrid(1,1,1); // 1 block
    dim3 dimBlock(4,4,1); // 16 threads


    // launch kernel
    MatAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,N,N);
    cudaDeviceSynchronize();


    // transfer result from device to host
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);


    // print results
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
        
    }


    // free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);




    return 0;
}