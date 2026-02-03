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

    print("Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    thr_coord = (tidx, None)
    # thr_coord = (tidx, cute.repeat_like(None, gA.shape[1]))

    # slice for threads: vid -> address
    thrA = tidfrgA[thr_coord]  # (V) -> physical address
    thrB = tidfrgB[thr_coord]  # (V) -> physical address
    thrC = tidfrgC[thr_coord]  # (V) -> physical address

    thrC[None] = thrA.load() + thrB.load()


