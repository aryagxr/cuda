// simple cuda test

#include <iostream>

__global__ void helloCUDA() {
  printf("Hello World from CUDA!\n");
}

int main() {
  helloCUDA<<<1, 1>>>(); // 1 block, 1 thread
  cudaDeviceSynchronize();
  return 0;
}