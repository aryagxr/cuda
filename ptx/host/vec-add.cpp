
#include <cuda.h>
#include <iostream>
#include <vector>

int main() {

    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    

    CUmodule module;
    cuModuleLoad(&module, "/home/ari/cuda-build/ptx/vec-add.cubin"); //bin file
    

    CUfunction function;
    cuModuleGetFunction(&function, module, "vector_add");
    

    const int N = 10000;
    std::vector<float> h_A(N, 1.0f);  //  1.0
    std::vector<float> h_B(N, 2.0f);  //  2.0
    std::vector<float> h_C(N);        
    

    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, N * sizeof(float));
    cuMemAlloc(&d_B, N * sizeof(float));
    cuMemAlloc(&d_C, N * sizeof(float));

    cuMemcpyHtoD(d_A, h_A.data(), N * sizeof(float));
    cuMemcpyHtoD(d_B, h_B.data(), N * sizeof(float));
    
    int N_copy = N;
    void* args[] = { &d_A, &d_B, &d_C, &N_copy };

    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop, CU_EVENT_DEFAULT);
    
    // Warmup run
    cuLaunchKernel(function,
                  (N + 255) / 256, 1, 1,    
                   256, 1, 1,               
                   0, nullptr,              
                   args, nullptr);          
    cuCtxSynchronize();
    

    cuEventRecord(start, 0);
    
    cuLaunchKernel(function,
                  (N + 255) / 256, 1, 1,    
                   256, 1, 1,               
                   0, nullptr,              
                   args, nullptr);          
    
    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);
    

    float milliseconds = 0;
    cuEventElapsedTime(&milliseconds, start, stop);
    
    

    cuMemcpyDtoH(h_C.data(), d_C, N * sizeof(float));

    std::cout << "PTX Vector addition performance:" << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    

    std::cout << "\nVector addition results (first 10 elements):" << std::endl;
    for (int i = 0; i < 10 && i < N; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }
    

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);

    cuEventDestroy(start);
    cuEventDestroy(stop);
    

    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}