// check available resources


#include <stdio.h>
#include <cuda_runtime.h>


int main(){

    int dev_count;
    cudaGetDeviceCount(&dev_count);

    cudaDeviceProp dev_prop;
    for(int i=0; i< dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
    }

    printf("CUDA enabled devices in the system: %d\n", dev_count);
    printf("Compute capability: %d.%d\n", dev_prop.major, dev_prop.minor);

    printf("Threads allowed in Block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("Number of SMs in the device: %d\n", dev_prop.multiProcessorCount);
    printf("Clock frequency of device: %d\n", dev_prop.clockRate);

    printf("Max number of threads allowed in X dimension block: %d\n", dev_prop.maxThreadsDim[0]);
    printf("Max number of threads allowed in Y dimension block: %d\n", dev_prop.maxThreadsDim[1]);
    printf("Max number of threads allowed in Z dimension block: %d\n", dev_prop.maxThreadsDim[2]);

}