import torch
import torch.nn as nn
import time

m,n = 1024,1024

# input tensor is 1,2,3,4...16
input = torch.arange(1, m*n+1).reshape(m,n).float()

# LayerNorm
layer_norm = nn.LayerNorm(n, elementwise_affine=False, eps=1e-6).cuda()

#warm up
for i in range(10):
    output = layer_norm(input.cuda())


# measure
start = time.time()
for i in range(1000):
    output = layer_norm(input.cuda())
torch.cuda.synchronize()
end = time.time()

pytorch_time = (end - start)/1000
print(f"PyTorch LayerNorm time: {pytorch_time * 1000:.4f} ms")
print(input)
print(output)