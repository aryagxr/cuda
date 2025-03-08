#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buffer_len = 1024;

__global__ void transfer_1d_array(int* data, size_t offset){

    // define shared mem 16 byte aligned
    __shared__ alignas(16) int smem[buffer_len];

    
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar; // variable in shared memory

    // initializing smem barrier
    // done once per block
    if(threadIdx.x == 0){ 

        // init barrier for all threads in a block
        init(&bar, blockDim.x);

        // create a memory fence
        // ensures the block's reads & writes to smem are complete
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();


    // Initiate TMA transfer from global to shared mem
    if(threadIdx.x == 0){

        // arrive at smem barier & get byte count
        cuda::memcpy_async(
            smem, 
            data + offset,
            cuda::aligned_size_t<16>(sizeof(smem)),
            bar
        );
    }

    // all threads arrive at the barrier
    barrier::arrival_token token = bar.arrive();


}




