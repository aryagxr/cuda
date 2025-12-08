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
        result = fn(*args)
    torch.cuda.synchronize()

    # timing
    start = time.time()
    for _ in range(iters):
        result = fn(*args)
    torch.cuda.synchronize()

    dt = (time.time() - start) / iters
    tflops = FLOPs / (dt * 1e12)
    return dt * 1000, tflops, result


print("Computing reference result...")
ref = torch.matmul(x, w)
print("Done.\n")

print("=== Correctness Check ===")
results = {}

kernels = [
    ("naive_fp32", matmul.matmul_naive_fp32, xf32, wf32),
    ("naive_bf16", matmul.matmul_naive_bf16, x, w),
    ("smem_bf16", matmul.matmul_smem_bf16, x, w),
    ("blocktiling_1d", matmul.blocktiling_1d, x, w),
    ("MMA_tc_ldmatrix", matmul.matmul_mma_bf16, x, w),
    ("torch_bmm", torch.matmul, x, w),
]

for name, fn, *args in kernels:
    ms, tflops, result = bench(fn, *args)
    results[name] = (ms, tflops)
    
    diff = torch.abs(result.float() - ref.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if max_diff <= 1.0: 
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"{name:15s}  {status}  (max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f})")

print("\n=== Benchmark Results ===")
for name, (ms, tflops) in results.items():
    print(f"{name:15s}  {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")