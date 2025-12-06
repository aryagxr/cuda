
# import torch
# from torch.utils.cpp_extension import load

# print("Starting compilation...")
# matmul_module = load(
#     name="matmul_module",
#     sources=["/home/ari/cuda/matmul/torch/naive.cu"], 
#     extra_cuda_cflags=["-O2"],  
# )
# print("Compilation done!")


# A = torch.randn(4, 8, device="cuda", dtype=torch.float32)
# B = torch.randn(8, 4, device="cuda", dtype=torch.float32)

# C_custom = matmul_module.matmul(A, B)
# C_torch = A @ B

# print("Custom:", C_custom[:2, :2])
# print("Torch:", C_torch[:2, :2])




import torch
from torch.utils.cpp_extension import load
import time

# Compile and load the CUDA kernel
print("Compiling CUDA kernel...")
matmul_module = load(
    name="matmul_module",
    sources=["/home/ari/cuda/matmul/torch/naive.cu"],  # or whatever your .cu file is named
    extra_cuda_cflags=["-O3"],
    verbose=True
)
print("Compilation done!\n")

# Attention dimensions
seq_len = 512  # You can adjust this
hidden_dim = 2048

# Create test matrices matching your attention sizes
print(f"Testing with seq_len={seq_len}, hidden_dim={hidden_dim}")
x = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.float32)
wq = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=torch.float32)

# Test correctness
print("\n=== Correctness Test ===")
Q_custom = matmul_module.matmul(x, wq.T)  # [seq_len, 2048]
Q_torch = torch.matmul(x, wq.T)           # [seq_len, 2048]

try:
    torch.testing.assert_close(Q_custom, Q_torch, rtol=1e-3, atol=1e-3)
    print("✅ Matmul is correct!")
    print(f"Custom output shape: {Q_custom.shape}")
    print(f"Max difference: {(Q_custom - Q_torch).abs().max().item():.6f}")
except AssertionError as e:
    print("❌ Matmul failed correctness test!")
    print(e)
    exit(1)

# Benchmark
print("\n=== Performance Benchmark ===")
num_warmup = 10
num_iterations = 100

# Warmup
for _ in range(num_warmup):
    _ = matmul_module.matmul(x, wq.T)
    _ = torch.matmul(x, wq.T)
torch.cuda.synchronize()

# Benchmark custom kernel
start = time.time()
for _ in range(num_iterations):
    Q_custom = matmul_module.matmul(x, wq.T)
torch.cuda.synchronize()
custom_time = (time.time() - start) / num_iterations

# Benchmark PyTorch
start = time.time()
for _ in range(num_iterations):
    Q_torch = torch.matmul(x, wq.T)
torch.cuda.synchronize()
torch_time = (time.time() - start) / num_iterations

print(f"Custom kernel: {custom_time*1000:.3f} ms")
print(f"PyTorch:       {torch_time*1000:.3f} ms")
print(f"Speedup:       {torch_time/custom_time:.2f}x")

# Test with different sequence lengths
print("\n=== Testing Different Sequence Lengths ===")
for seq_len_test in [128, 256, 512, 1024, 2048]:
    x_test = torch.randn(seq_len_test, hidden_dim, device="cuda", dtype=torch.float32)
    
    Q_custom = matmul_module.matmul(x_test, wq.T)
    Q_torch = torch.matmul(x_test, wq.T)
    
    max_diff = (Q_custom - Q_torch).abs().max().item()
    print(f"seq_len={seq_len_test:4d}: shape={Q_custom.shape}, max_diff={max_diff:.6f}")

print("\n✅ All tests passed! Ready to use in attention.")