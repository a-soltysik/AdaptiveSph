#include <cuda_runtime_api.h>
#include <cstdio>

__global__ void helloWorldKernel()
{
    printf("Hello World!\n");
}

extern void helloWorld()
{
    helloWorldKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
