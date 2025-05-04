#pragma once

namespace sph::cuda::device
{

__device__ auto densityKernel(float distance, float smoothingRadius) -> float;
__device__ auto nearDensityKernel(float distance, float smoothingRadius) -> float;
__device__ auto densityDerivativeKernel(float distance, float smoothingRadius) -> float;
__device__ auto nearDensityDerivativeKernel(float distance, float smoothingRadius) -> float;
__device__ auto viscosityKernel(float distance, float smoothingRadius) -> float;
__device__ auto viscosityLaplacianKernel(float distance, float smoothingRadius) -> float;
}
