import cutlass.cute as cute
import torch

@cute.kernel
def add_vectorized_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    #global thread idx
    gtidx = tidx + (bidx * bdimx)

    m, n = gA.shape[1] #rows, col blocks = 2048, 512
    #map gtidx to row, col indices of output tensor
    ni = gtidx % n
    mi = gtidx // n

    #load A and B as vectorized loads
    avals = gA[(None, (mi, ni))].load()
    bvals = gB[(None, (mi, ni))].load()


    #add
    cvals = avals + bvals

    #store result
    gC[(None, (mi, ni))] = cvals



@cute.jit
def add_vect_host(hA, hB, hC):
    threads_per_block = 256

    #cute zip the tensors in host size
    gA = cute.zipped_divide(hA, (1, 4))
    gB = cute.zipped_divide(hB, (1, 4))
    gC = cute.zipped_divide(hC, (1, 4))

    

    #launch kernel with vectorized params
    add_vectorized_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )



M, N = 16384, 8192
#fp16 data on device
A = torch.rand(M, N, dtype=torch.float16, device="cuda")
B = torch.rand(M, N, dtype=torch.float16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float16, device="cuda")


A = A.contiguous()
B = B.contiguous()
C = C.contiguous()

add_vect_host(A, B, C)

#verify
torch.testing.assert_close(C, A + B)
