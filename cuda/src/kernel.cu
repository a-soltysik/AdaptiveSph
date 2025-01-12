#include <cuda_runtime_api.h>

#include <cstdio>

#include "cuda/kernel.cuh"

__global__ void helloWorldKernel()
{
    printf("Hello World!\n");
}

void helloWorld()
{
    helloWorldKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
