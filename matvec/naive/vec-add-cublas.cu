// Vector addition using cuBLAS

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cassert>


// initialize vector elements
void vec_elements(float *a, int n){
    for(int i=0; i<n; i++){
        a[i] = (float)(rand() % 100);
    }
}

// checking result
void res(float *a, float*b, float *c, float scalar, int n){
    printf("Last five elements comparison:\n");
    printf("Index   A[i]    B[i]    (%.1f * A[i]) + B[i]    C[i]\n", scalar);
    printf("------------------------------------------------------\n");
    
    for(int i = n - 5; i < n; i++){  // Print last 5 elements
        float expected = scalar * a[i] + b[i];
        printf("%4d  %6.1f  %6.1f  %10.1f            %6.1f\n", 
               i, a[i], b[i], expected, c[i]);
        assert(c[i] == expected);
    }
}

int main(){

    // initialize vector size
    const int n = 1000;
    size_t vector_size = n*sizeof(float);
    const float alpha = 2.0f;

    // allocate host memory
    float *h_A = (float*)malloc(vector_size);
    float *h_B = (float*)malloc(vector_size);
    float *h_C = (float*)malloc(vector_size);

    // init device variables
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, vector_size);
    cudaMalloc((void**)&d_B, vector_size);

    // init vectors
    vec_elements(h_A, n);
    vec_elements(h_B, n);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);


    // copy vectors from host to device the cublas way
    cublasSetVector(n, sizeof(float), h_A, 1, d_A, 1); // step size 1
    cublasSetVector(n, sizeof(float), h_B, 1, d_B, 1);

    // C[i] = alpha*A[i] + B[i]
    // we don't need a kernel, just call this cublas function
    cublasSaxpy(handle, n, &alpha, d_A, 1, d_B, 1);

    // instead of memcpy device to host
    cublasGetVector(n, sizeof(float), d_B, 1, h_C, 1);

    // print results
    res(h_A, h_B, h_C, alpha, n);

    // clean up handle
    cublasDestroy(handle);

    // free memory
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B); free(h_C);


}