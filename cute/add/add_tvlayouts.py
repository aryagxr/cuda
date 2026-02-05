#taken from https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb

import cutlass
import cutlass.cute as cute
import torch

#figure out using dlpack

@cute.kernel
def add_kernel_tvlayouts(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tvlayout: cute.Layout):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    #gA((TileM, TileN):(BlockM, BlockN))
    #take entire tile
    block_coords = ((None, None), bidx)
    blckA = gA[block_coords]
    blckB = gB[block_coords]
    blckC = gC[block_coords]

    #composition
    tidfragA = cute.composition(blckA, tvlayout)
    tidfragB = cute.composition(blckB, tvlayout)
    tidfragC = cute.composition(blckC, tvlayout)

    thr_coord = (tidx, None)
    # thr_coord = (tidx, cute.repeat_like(None, gA.shape[1]))

    # slice for threads: vid -> address
    thrA = tidfragA[thr_coord]  # (V) -> physical address
    thrB = tidfragB[thr_coord]  # (V) -> physical address
    thrC = tidfragC[thr_coord]  # (V) -> physical address

    thrC[None] = thrA.load() + thrB.load()




@cute.jit
def elementwise_add(mA, mB, mC):
    # PyTorch tensors will be automatically converted to CUTE tensors
    coalesced_ldst_bytes = 16

    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print("Tiled Input Tensors:")
    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")


    kernel = add_kernel_tvlayouts(gA, gB, gC, tv_layout)
    kernel.launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
    
    torch.cuda.synchronize()
    print("[DSL INFO] Kernel execution completed")


M, N = 16384, 8192
#fp16 data on device
A = torch.rand(M, N, dtype=torch.float16, device="cuda")
B = torch.rand(M, N, dtype=torch.float16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

A = A.contiguous()
B = B.contiguous()
C = C.contiguous()

elementwise_add(A, B, C)


torch.testing.assert_close(C, A + B)
