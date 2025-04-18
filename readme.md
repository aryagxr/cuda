**CUDA Progress**

| **Day**    | **Code Summary**                                                   |
|------------|--------------------------------------------------------------------|
| Day 1      |  CUDA set up and kernel that prints "Hello World"                  |
| Day 2      |  CUDA kernel that adds two vectors                                 |
| Day 3      |  Adding matrices                                                   |
| Day 4      |  Vector addition using cuBLAS                                      |
| Day 5      |  Naive matmul                                                      |
| Day 6      |  Tiled matmul using shared memory                                  |
| Day 7      |  Naive 1D convolution with boundary checks                         |
| Day 8      |  Matrix multiplication using cuBLAS                                |
| Day 9      |  Matrix Transpose                                                  |
| Day 10 🥳  |  Naive Softmax                                                     |
| Day 11     |  Softmax using shared memory and reductions                        |
| Day 12     |  Softmax using warp shuffle functions                              |
| Day 13     |  1D complex-to-complex fourier transform using cuFFT               |
| Day 14     |  Naive layer normalization                                         |
| Day 15     |  Optimizing layer norm using shared memory                         |
| Day 16     |  Optimizing layer norm using warp shuffle functions                |
| Day 17     |  Optimizing layer norm using vectorized loads                      |
| Day 18     |  Tiled 1D convolution and halo cells                               |
| Day 19     |  1D convolution using L2 cache                                     |
| Day 20 🥳  |  [Blog Post: Optimizing Layer Normalization with CUDA](https://aryagxr.com/blogs/cuda-optimizing-layernorm) |
| Day 21     |  Simple self attention                                             |
| Day 22     |  Optimizing self attention                                         |
| Day 23     |  Causal attention with masking                                     |
| Day 24     |  Causal attention + torch binding                                  |
| Day 25     |  Multi-head attention                                              |
| Day 26     |  Parallel add using koggle stone algorithm                         |
| Day 27     |  MHA debug                                                         |
| Day 28     |  Flash Attention 1 (algorithm 1) Forward pass                      |
| Day 29     |  Flash Attention 1 (algorithm 1) Forward pass continued            |
| Day 30 🥳  |  Flash Attention 1 (algorithm 1) Forward pass                      |
| Day 31     |  HGEMV matvec using fp16                                           |
| Day 32     |  HGEMV matvec using Bfloat16                                       |
| Day 33     |  Matmul using Tensor cores                                         |
| Day 34     |  Swizzle patterns on matrix transpose                              |
| Day 35     |  Swizzled matrix transpose using Tensor Memory Accelerators        |
| Day 36     |  Brent Kung Parallel Scan algorithm                                |
| Day 37     |  Matvec using integer fixed point arithmetic                       |
| Day 38     |  Transfered 1D array from gmem->smem->gmem using TMA               |
| Day 39     |  Memory Coalesced layernorm + revisited Flash attention            |
| Day 40 🥳  |  revisited Flash Attention 1                                       |
| Day 41     |  Flash Attention 1                                                 |
| Day 42     |  Flash Attention 1                                                 |
| Day 43     |  ReLU Activation - FP32, FP32x4, FP16, FP16x2 vectorized           |
| Day 44     |  Overlapping data transfers using CUDA Streams (Vector add)        |
| Day 45     |  ReLU using CUDA Streams + benchmarked                             |
| Day 46     |  Packed 128 bit ReLU FP16x8 kernel                                 |
| Day 47     |  Sparse matrix-vector mul (spMV)                                   |
| Day 48     |  Sparse padded matrix-vector mul                                   |
| Day 49     |  RoPE Kernel: Rotary Position Embedding naive fp32                 |
| Day 50 🥳  |  Optimized RoPE using vectorized loads and half precision (18x)    |
| Day 51     |  Flash Attention 2 Forward                                         |
| Day 52     |  Flash Attention 2 Forward                                         |
| Day 53     |  Flash Attention 2 Forward                                         |
| Day 54     |  Gaussian Elimination                                              |
| Day 55     |  PTX vector add kernel                                             |
| Day 56     |  GELU activatation naive fp32 kernel                               |
| Day 57     |  GELU activation vectorized                                        |
| Day 58     |  Backward pass kernel for Relu activation                          |
| Day 59     |  Backward pass kernel for GELU activation                          |
| Day 60 🥳  |  LeetGPU challenge - reduction                                     |
| Day 61     |  Optimize + benchmarked gelu kernels                               |
| Day 62     |  Micrograd in CUDA                                                 |
| Day 63     |  Micrograd in CUDA                                                 |
| Day 64     |  Micrograd in CUDA                                                 |
| Day 65     |  Micrograd in CUDA                                                 |
| Day 66     |  Optimized Sigmoid activation                                      |
| Day 67 - Day 70  🥳  |  Micrograd in CUDA                                       |
| Day 71     |  Sigmoid with half precision                                       |
| Day 72     |  Sigmoid with fp16 vectorized                                      |
| Day 73     |  Swish kernel                                                      |
| Day 74     |  Swish kernel vectorized                                           |
| Day 75     |  AMD hip kernel intro + vector add kernel                          |
| Day 76     |  Revisiting gemm optimizations                                     |
| Day 77     |  Gemm coalesced                                                    |
| Day 78     |  fp16 swish                                                        |
| Day 79     |  AMD competition fp8 gemm & swish optimizations                    |
| Day 80 🥳  |  AMD competition fp8 gemm optimizations                            |
