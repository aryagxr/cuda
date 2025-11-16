import cutlass
import cutlass.cute as cute

@cute.kernel
def hello_kernel():
    # cute.arch returns a 3-tuple (thread_x, thread_y, thread_z)
    tidx, tidy, tidz = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello World! from my GPU :)")

@cute.jit
def hello_world():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))

compiled = cute.compile(hello_world)
compiled()
