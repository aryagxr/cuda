#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

__global__ void SpMV_CSR(int rows, float *data, int *col_idx, int *row_ptr, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float dot = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row+1];
        
        for (int i = row_start; i < row_end; i++) {
            dot += data[i] * x[col_idx[i]];
        }
        
        y[row] += dot;
    }
}


int main() {
    const int rows = 4;
    const int nnz = 9; // number of non-zero elements

    float h_data[nnz] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    int h_col_idx[nnz] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int h_row_ptr[rows + 1] = {0, 3, 6, 7, 9};

    float h_x[3] = {1, 2, 3};
    float h_y[rows] = {0, 0, 0, 0};

    float *d_data, *d_x, *d_y;
    int *d_col_idx, *d_row_ptr;
    cudaMalloc(&d_data, nnz * sizeof(float));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_x, 3 * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));

    cudaMemcpy(d_data, h_data, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, rows * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    SpMV_CSR<<<numBlocks, blockSize>>>(rows, d_data, d_col_idx, d_row_ptr, d_x, d_y);

    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result vector y:" << std::endl;
    for (int i = 0; i < rows; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_data);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}