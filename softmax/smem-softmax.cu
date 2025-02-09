#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#define M 1024
#define N 32768

__global__ void smem_softmax(float *X, float *P, int m, int n){

    __shared__ float smem[1024];

    // one block for one row
    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if(row >= m){
        return;
    }

    float *row_in = X + row * n;
    float *row_out = P + row * n;

    float lmax = -1 * INFINITY;
    float lnorm = 0.0f;

    // computing local max and local norm
    for(int i = tidx; i<n; i += blockDim.x){
        float x = row_in[i];

        if(x > lmax){
            lnorm *= expf(lmax-x);
            lmax = x;
        }
        lnorm += expf(x-lmax);
    }

    __syncthreads();

    // thread that computes the local max of its set of blockdim spaced elements
    // saves that local max in that corresponding thread index in smem
    smem[tidx] = lmax;
    __syncthreads();


    // computing global max
    for(int stride = blockDim.x/2; stride > 0; stride /=2){
        if(tidx < stride){
            smem[tidx] = max(smem[tidx], smem[tidx + stride]);
        }

        __syncthreads();
    }

    // global max of that row
    float gmax = smem[0];
    __syncthreads();

    // global norm
    for(int stride = blockDim.x/2; stride > 0; stride >>=1){
        if(tidx < stride){
            smem[tidx] += smem[tidx+stride];
        }
        __syncthreads();
    }

    float gnorm = smem[0];
    __syncthreads();


    // computing softmax
    for(int i=tidx; i<n; i+=blockDim.x){
        row_out[i]= expf(row_in[i] - gmax) / gnorm;
    }

}


int main(){

    size_t matrix_size = M*N*sizeof(float);

    float *X_input, *P_output;
    float *D_input, *D_output;

    X_input = (float*)malloc(matrix_size);
    P_output = (float*)malloc(matrix_size);

    cudaMalloc((void**)&D_input, matrix_size);
    cudaMalloc((void**)&D_output, matrix_size);

    for(int a = 0; a<M*N; a++){
        X_input[a] = ((float)rand() / RAND_MAX) * 10.0f;
    }

    cudaMemcpy(D_input, X_input, matrix_size, cudaMemcpyHostToDevice);


    
    dim3 ThreadsPerBlock(256);
    dim3 blocksPerGrid(M); // m rows, m blocks

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    smem_softmax<<<blocksPerGrid, ThreadsPerBlock>>>(D_input, D_output, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(P_output, D_output, matrix_size, cudaMemcpyDeviceToHost);

    printf("Softmax result (first row only):\n");
    for (int j = 0; j < 10; j++) { 
        printf("%f ", P_output[j]);
    }
    printf("...\n");

    cudaFree(D_input); cudaFree(D_output);
    free(X_input); free(P_output);

    return 0;

}