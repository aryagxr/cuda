import cutlass.cute as cute
import torch
from add_naive import add_host
from add_vectorized import add_vect_host

M, N = 16384, 8192
A = torch.rand(M, N, dtype=torch.float16, device="cuda")
B = torch.rand(M, N, dtype=torch.float16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

num_elements = sum([A.numel(), B.numel(), C.numel()])

A = A.contiguous()
B = B.contiguous()
C = C.contiguous()

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
    
    return avg_time_us, achieved_bandwidth

results = []

print("Benchmarking naive kernel...")
C_naive = torch.zeros_like(C)
time_naive, bw_naive = benchmark(add_host, A, B, C_naive)
results.append(("Naive", time_naive, bw_naive))

print("Benchmarking vectorized kernel...")
C_vect = torch.zeros_like(C)
time_vect, bw_vect = benchmark(add_vect_host, A, B, C_vect)
results.append(("Vectorized", time_vect, bw_vect))

print("\n" + "="*70)
print(f"{'Kernel':<20} {'Time (us)':<15} {'Bandwidth (GB/s)':<20} {'Speedup':<10}")
print("="*70)

baseline_time = results[0][1]
for name, time, bw in results:
    speedup = baseline_time / time
    print(f"{name:<20} {time:<15.4f} {bw:<20.2f} {speedup:<10.2f}x")
print("="*70)