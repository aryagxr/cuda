import torch
from torch.utils.cpp_extension import load
import time

print("Compiling CUDA...")
matmul = load(
    name="matmul_module",
    sources=["/home/ari/cuda/matmul/gemm/kernels.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=False
)
print("Done.\n")


M = 512
K = 2048
N = 2048
FLOPs = 2 * M * K * N

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
xf32 = x.float()
wf32 = w.float()

def bench(fn, *args, iters=50):
    # warmup
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()

    # timing
    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()

    dt = (time.time() - start) / iters
    tflops = FLOPs / (dt * 1e12)
    return dt * 1000, tflops


results = {}

results["naive_fp32"] = bench(matmul.matmul_naive_fp32, xf32, wf32)
results["naive_bf16"] = bench(matmul.matmul_naive_bf16, x, w)
results["smem_bf16"]  = bench(matmul.matmul_smem_bf16, x, w)
results["torch_bmm"]  = bench(torch.matmul, x, w)

print("\n=== Benchmark Results ===")
for name, (ms, tflops) in results.items():
    print(f"{name:15s}  {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")
