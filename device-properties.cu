#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU name: " << prop.name << "\n";
    std::cout << "Compute capability: " 
              << prop.major << "." << prop.minor << "\n";

    std::cout << "Multiprocessors (SMs): " 
              << prop.multiProcessorCount << "\n";

    std::cout << "Max threads per block: " 
              << prop.maxThreadsPerBlock << "\n";

    std::cout << "Max threads per SM: " 
              << prop.maxThreadsPerMultiProcessor << "\n";

    std::cout << "Warp size: " 
              << prop.warpSize << "\n";

    std::cout << "Max warps per SM: "
              << prop.maxThreadsPerMultiProcessor / prop.warpSize << "\n";

    std::cout << "Registers per block: "
              << prop.regsPerBlock << "\n";

    std::cout << "Registers per SM: "
              << prop.regsPerMultiprocessor << "\n";

    std::cout << "Shared mem per block: "
              << prop.sharedMemPerBlock / 1024 << " KB\n";

    std::cout << "Shared mem per SM: "
              << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";

    return 0;
}
