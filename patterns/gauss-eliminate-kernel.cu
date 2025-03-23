
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 8
#define BLOCK_SIZE 8


/* Kernel to normalize the current row */
__global__ void division_kernel(float *U, float *current_row, int k) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < MATRIX_SIZE - k) {
        int col = k + tidx;
        float pivot = U[k * MATRIX_SIZE + k];  
        U[k * MATRIX_SIZE + col] /= pivot;  
        current_row[tidx] = U[k * MATRIX_SIZE + col];  
    }
}

/* Kernel to eliminate values below the diagonal */
__global__ void elimination_kernel(float *U, float *current_row, int k) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int row = k + 1 + tidy; 
    int col = k + tidx;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        float factor = U[row * MATRIX_SIZE + k];  
        U[row * MATRIX_SIZE + col] -= factor * current_row[col - k];
    }
}

void initialize_matrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (rand() % 10) + 1;
    }
}

void print_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%6.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gaussian_elimination(float *U) {
    float *d_U, *d_current_row;
    cudaMalloc(&d_U, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc(&d_current_row, MATRIX_SIZE * sizeof(float));

    cudaMemcpy(d_U, U, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int k = 0; k < MATRIX_SIZE; k++) {
        division_kernel<<<(MATRIX_SIZE - k + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_U, d_current_row, k);
        cudaDeviceSynchronize();

        elimination_kernel<<<gridDim, blockDim>>>(d_U, d_current_row, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(U, d_U, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_U);
    cudaFree(d_current_row);
}

int main() {
    srand(42);
    float *matrix = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    initialize_matrix(matrix, MATRIX_SIZE);
    printf("Original Matrix:\n");
    print_matrix(matrix, MATRIX_SIZE);

    gaussian_elimination(matrix);

    printf("Transformed Matrix (Upper Triangular Form):\n");
    print_matrix(matrix, MATRIX_SIZE);

    free(matrix);
    return 0;
}


