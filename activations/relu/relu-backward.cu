#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>


// deriv = 1 for x>=0, 0 for x<0
__global__ void relu_backpass(float* in, float* din, float* dout, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < n){
        float x = in[tidx];
        din[tidx] = (x > 0) ? dout[tidx] : 0.0f;
    }

}