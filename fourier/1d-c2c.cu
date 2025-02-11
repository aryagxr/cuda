#include <complex.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cufft.h>

void runFFT(cufftHandle &plan, cudaStream_t &stream, std::vector<std::complex<float>>&signal, int fft_size, int batch){

    cufftComplex *gpu_data = nullptr;

    // malloc
    cudaMalloc(reinterpret_cast<void **>(&gpu_data), sizeof(std::complex<float>) * signal.size());
    cudaMemcpyAsync(gpu_data, signal.data(), sizeof(std::complex<float>) * signal.size(), cudaMemcpyHostToDevice, stream);

    // forward fft: transforms time doimain input to frequency domain
    cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);


    cudaMemcpyAsync(signal.data(), gpu_data, sizeof(std::complex<float>) * signal.size(), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(gpu_data);
}



int main(){
    
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int fft_size = 8;
    int batch = 2;
    int num_elements = batch * fft_size;

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    // initializing complex values
    std::vector<data_type> signal(num_elements, 0);

    for(int i=0; i<num_elements; i++){
        signal[i] = data_type(i, -i);
    }

    printf("Input array:\n");
    for(auto &i : signal){
        printf("%f + %fj\n", i.real(), i.imag());
    }


    // we need a cuFFT plan
    cufftCreate(&plan);
    cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch);

    // set up cuda stream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, stream);

    runFFT(plan, stream, signal, fft_size, batch);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    printf("Output: \n");
    for(auto &i : signal){
        printf("%f + %fj\n", i.real(), i.imag());
    }


    // destroy handle
    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaDeviceReset();

    return 0;

}



