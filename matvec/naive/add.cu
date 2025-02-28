// vector addition

#include <iostream>
#include <cuda_runtime.h>


// kernel definition
__global__ void VecAdd(const int *A, int *B, int *C, int N){

    // thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<N){

        // vector addition computation
        C[idx] = A[idx] + B[idx];
    }
   
}


// main
int main(){

    // initialize number of bytes to allocate for N ()
    const int N = 1024; // each vector has N elements
    const int vector_size = N*sizeof(int); //memory needed for each vector
    

    // allocate host memory (cpu)
    int *h_A = (int*)malloc(vector_size);
    int *h_B = (int*)malloc(vector_size);
    int *h_C = (int*)malloc(vector_size);


    // allocate device memory (gpu)
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, vector_size);
    cudaMalloc(&d_B, vector_size);
    cudaMalloc(&d_C, vector_size);


    // initialize the vector elements
    for(int i=0; i<N; i++){
        h_A[i] = i;
        h_B[i] = 1;
    }


    // transfer memory from host to device
    // transfer only A and B since C is to be computed by gpu
    cudaMemcpy(d_A, h_A, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vector_size, cudaMemcpyHostToDevice);


    // define block and grid dim
    int threadsInBlock = 256;
    int blocksInGrid = (N + threadsInBlock-1) / threadsInBlock;
    //int blocksInGrid = 1;
    

    // launch kernel
    VecAdd<<<blocksInGrid, threadsInBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();


    // transfer result array memory from device back to host
    cudaMemcpy(h_C, d_C, vector_size, cudaMemcpyDeviceToHost);

    for(int a = N-10; a<N; a++){
        std::cout << "C[" << a << "] = " << h_C[a] << std::endl;
        //std::cout << "B[" << a << "] = " << h_B[a] << std::endl;
    }


    // free allocated device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}