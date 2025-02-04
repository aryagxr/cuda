// Simple 1D convolution

#include <iostream>
#include <cuda_runtime.h>

__global__ void conv1d(float *N, float *M, float *P, int mask_width, int width){

    // 1D output element index
    int i = (blockDim.x*blockIdx.x) + threadIdx.x;
    float P_val = 0;
    int N_start = i - (mask_width / 2);

    // iterate over mask length and compute dot product
    for(int j=0; j<mask_width; j++){

        // boundary check
        if(N_start + j>=0 && N_start + j < width){
            P_val += N[N_start + j]*M[j];
        }
    }
    P[i] = P_val;

}


int main(){

    const int input_size = 15;
    const int mask_size = 5;

    float N[input_size] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float M[mask_size] = {0.1, 0.2, 0.3, 0.2, 0.1};

    float P[input_size];
    float *d_N, *d_M, *d_P; //device variables

    cudaMalloc((void**)&d_N, input_size*sizeof(float));
    cudaMalloc((void**)&d_M, mask_size*sizeof(float));
    cudaMalloc((void**)&d_P, input_size*sizeof(float)); //output same size as input

    cudaMemcpy(d_N, N, input_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_size*sizeof(float), cudaMemcpyHostToDevice);

    // kernel size 1d
    // 15 threads needed
    int threadsPerBlock = 64;
    int blocksPerGrid = (input_size+threadsPerBlock -1)/threadsPerBlock; // 1 block needed

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    conv1d<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, mask_size, input_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Ensure kernel execution is finished

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    cudaMemcpy(P, d_P, input_size*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Convolution output: ");
    for(int i=0; i<input_size;i++){
        printf("%f ", P[i]);
    }
    

    cudaFree(d_M);cudaFree(d_N);cudaFree(d_P);

    return 0;
}