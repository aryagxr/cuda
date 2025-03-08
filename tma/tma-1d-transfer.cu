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


    // all threads signal that it arrive at the barrier
    // threads return the arrival token
    barrier::arrival_token token = bar.arrive();

    // wait for data to arrive
    bar.wait(std::move(token));

    // compute saxpy (single prec ax+y) in smem
    for(int i = threadIdx.x; i < buffer_len; i += blockDim.x){
        smem[i] += 1; // just adding 1 to elements in smem
    }

    __syncthreads();

    // wait for smem writes to be visible
    // order the smem writes before next step
    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();


    // initiate tma transfer from smem back to gmem
    if(threadIdx.x == 0){
        ptx::cp_async_bulk(
            ptx::space_global,
            ptx::space_shared,
            data + offset, smem, sizeof(smem)
        );

        // commit & wait for tma to finish reading from smem
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
    __syncthreads();

}


int main() {

    size_t array_size = 1024;
    size_t offset = 0;

    int *h_data, *d_data;
    size_t bytes = array_size * sizeof(int);

    // host memory
    h_data = new int[array_size];

    for (size_t i = 0; i < array_size; i++) {
        h_data[i] = i;
    }

    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);


    int threads_per_block = 256;
    int num_blocks = 4;
    transfer_1d_array<<<num_blocks, threads_per_block>>>(d_data, offset);


    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);


    bool success = true;
    for (size_t i = 0; i < array_size; i++) {
        if (h_data[i] != static_cast<int>(i + 1)) {
            std::cerr << "Mismatch at index " << i << ": expected " << i + 1 << ", got " << h_data[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }

    free(h_data);
    cudaFree(d_data);

    return 0;
}


