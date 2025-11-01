import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load

# compile, load and run .cu file

vectadd_module = torch.utils.cpp_extension.load(
    "vectadd_module",
    sources=["/home/ari/cuda/matvec/torch/vect_add.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)



# set values
a = torch.randint(1, 100, (1000,), device="cuda", dtype=torch.int32)
b = torch.randint(1, 100, (1000,), device="cuda", dtype=torch.int32)
c = vectadd_module.vect_add(a, b)

torch.testing.assert_close(c, a + b)

print("Input1:", a[:5])
print("Input2:", b[:5])
print("Output:", c[:5])