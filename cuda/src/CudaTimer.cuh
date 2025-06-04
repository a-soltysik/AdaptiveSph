#pragma once

namespace sph::benchmark
{
class CudaTimer
{
public:
    CudaTimer();
    ~CudaTimer();
    // Disable copy/move to avoid CUDA event management issues
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer(CudaTimer&&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    CudaTimer& operator=(CudaTimer&&) = delete;
    /**
     * Start timing measurement
     */
    void start();

    /**
     * Stop timing measurement and return elapsed time
     * @return Elapsed time in milliseconds
     */
    auto stop() -> float;

    /**
     * Check if timer is currently running
     */
    [[nodiscard]] auto isRunning() const -> bool
    {
        return _isRunning;
    }

private:
    cudaEvent_t _startEvent;
    cudaEvent_t _stopEvent;
    bool _isRunning = false;
    bool _initialized = false;
};
}
