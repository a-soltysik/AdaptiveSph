#include "CudaTimer.cuh"

namespace sph::benchmark
{

CudaTimer::CudaTimer()
{
    if (cudaEventCreate(&_startEvent) == cudaSuccess && cudaEventCreate(&_stopEvent) == cudaSuccess)
    {
        _initialized = true;
    }
    else
    {
        _initialized = false;
    }
}

CudaTimer::~CudaTimer()
{
    if (_initialized)
    {
        cudaEventDestroy(_startEvent);
        cudaEventDestroy(_stopEvent);
    }
}

void CudaTimer::start()
{
    if (!_initialized)
    {
        return;
    }
    if (_isRunning)
    {
        return;
    }
    cudaEventRecord(_startEvent);
    _isRunning = true;
}

auto CudaTimer::stop() -> float
{
    if (!_initialized || !_isRunning)
    {
        return 0.0F;
    }
    cudaEventRecord(_stopEvent);
    cudaEventSynchronize(_stopEvent);

    float elapsedTime = 0.0F;
    cudaEventElapsedTime(&elapsedTime, _startEvent, _stopEvent);

    _isRunning = false;
    return elapsedTime;  // Returns time in milliseconds
}

}
