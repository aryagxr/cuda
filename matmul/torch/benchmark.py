
import torch
from torch.utils.cpp_extension import load

print("Starting compilation...")
matmul_module = load(
    name="matmul_module",
    sources=["/home/ari/cuda/matmul/torch/naive.cu"], 
    extra_cuda_cflags=["-O2"],  
)
print("Compilation done!")


A = torch.randn(4, 8, device="cuda", dtype=torch.float32)
B = torch.randn(8, 4, device="cuda", dtype=torch.float32)

C_custom = matmul_module.matmul(A, B)
C_torch = A @ B

print("Custom:", C_custom[:2, :2])
print("Torch:", C_torch[:2, :2])