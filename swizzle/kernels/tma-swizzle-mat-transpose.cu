#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>



using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;


/*  Kernel 1: Transposing an 8*8 matrix of type int4
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