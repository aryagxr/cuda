#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>



using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;



/*  Kernel 1: Naive implementation of Matrix Transpose
    Using shared memory, with bank conflicts
*/
__global__ void kernel1_naive_smem_mattrans(float *A, float *A_T, int m, int n){

    __shared__ float smem[32][32];

    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < m && col < n){

        //load 32*32 tile into shared memory
        // each block has its own smem
        // blockdim = tile dim
        smem[threadIdx.x][threadIdx.y] = A[row * n + col];
        __syncthreads();

        // bank conflict happens here
        A_T[col * m + row] = smem[threadIdx.x][threadIdx.y];

    }
}


/*  Kernel 2: Matrix Transpose using shared memory & padding
    (without swizzling)
*/
__global__ void kernel2_smem_padding_mattrans(float *A, float *A_T, int m, int n){

    __shared__ float smem[32][33]; //extra padding col

    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < m && col < n){

        smem[threadIdx.x][threadIdx.y] = A[row * n + col];
        __syncthreads();
        A_T[col * m + row] = smem[threadIdx.x][threadIdx.y];

    }
}



/*  Kernel 3: Matrix Transpose with Swizzling
    XOR is used to swizzle smem indices
*/
__global__ void kernel3_swizzled_mattrans(float *A, float *A_T, int m, int n){

    __shared__ float smem[32][32];
    int row = threadIdx.y + (blockDim.y * blockIdx.y);
    int col = threadIdx.x + (blockDim.x * blockIdx.x);
    if(row < m && col < n){

        
        smem[threadIdx.x][threadIdx.x ^ threadIdx.y] = A[row * n + col];
        __syncthreads();

        A_T[col * m + row] = smem[threadIdx.x][threadIdx.x ^ threadIdx.y];

    }

}



/*  Kernel 4: Transposing an 8*8 matrix of type int4
    Using TMA instructions
    Sample code from CUDA documentation
*/

__global__ void kernel4_tma_int4_swizzle(const __grid_constant__ CUtensorMap tensor_map){

    __shared__ alignas(1024) int4 smem[8][8];
    __shared__ alignas(1024) int4 smem_T[8][8];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if(threadIdx.x == 0){
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token token;

    if(threadIdx.x == 0){
        // bulk tensor copy from gmem to smem
        cde::cp_async_bulk_tensor_2d_global_to_shared(&smem, &tensor_map, 0, 0, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
    }
    else{
        token = bar.arrive();
    }

    bar.wait(std::move(token));

    // matrix transpose swizzle
    for(int sidx_j = threadIdx.x; sidx_j < 8; sidx_j += blockDim.x){
        for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
            const int swiz_j_idx = (sidx_i % 8) ^ sidx_j; // XOR
            const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
            smem_T[sidx_j][swiz_i_idx_tr] = smem[sidx_i][swiz_j_idx];
        }
    }

    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    
    // unswizzle - write from smem to gmem
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, 0, 0, &smem_T);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }

    if (threadIdx.x == 0) {
        (&bar)->~barrier();
    }
    
}



int main(){
    const int M = 1024;
    const int N = 1024;
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


    dim3 threadsPerBlock(32,32); 
    dim3 blocksPerGrid(N/2, M/2); 

    kernel3_swizzled_mattrans<<<blocksPerGrid, threadsPerBlock>>>(dA, dA_t, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hA_t, dA_t, mat_size, cudaMemcpyDeviceToHost);

    /*
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 10; i++) {
            printf("%f ", hA_t[j * M + i]);
        }
        printf("\n");
    }
    printf("Successful");
    */

    cudaFree(dA); cudaFree(dA_t);
    free(hA); free(hA_t);

}