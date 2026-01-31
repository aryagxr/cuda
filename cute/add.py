import cutlass
import cutlass.cute as cute
import torch

@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    
    #get thread index
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    #set global thread index
    thread_idx = (bidx * bdimx) + tidx
    #tensor dim
    m, n = gA.shape
    #assign each thread to one element of the output tensor
    ni = thread_idx % n
    mi = thread_idx // n

    #load A and B
    aval = gA[mi, ni]
    bval = gB[mi, ni]
    
    gC[mi, ni] = aval + bval

    

@cute.jit
def add_host(hA, hB, hC):
    # PyTorch tensors will be automatically converted to CUTE tensors
    #configure launch parameters
    threads_per_block = 256
    m, n = hA.shape

    #create kernel instance
    kernel = add_kernel(hA, hB, hC)

    #launch kernel
    kernel.launch(
        grid=((m*n) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1)
    )

    


#test
M, N = 16384, 8192
#fp16 data on device
A = torch.rand(M, N, dtype=torch.float16, device="cuda")
B = torch.rand(M, N, dtype=torch.float16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

num_elements = sum([A.numel(), B.numel(), C.numel()])

A = A.contiguous()
B = B.contiguous()
C = C.contiguous()

add_host(A, B, C)

#verify
torch.testing.assert_close(C, A + B)


def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    bytes_per_element = a_.element_size()
    

    total_bytes = num_elements * bytes_per_element

    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")

benchmark(add_host, A, B, C)
