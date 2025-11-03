#include "FrameTimeManager.hpp"

#include <chrono>
#include <cstdint>

namespace sph::utils
{

auto FrameTimeManager::update() -> void
{
    const auto currentTime = std::chrono::steady_clock::now();
    _deltaTime = std::chrono::duration<float>(currentTime - _currentFrameStart).count();
    _currentFrameStart = currentTime;
    _frameCounter++;
}

auto FrameTimeManager::getDelta() const noexcept -> float
{
    return _deltaTime;
}

auto FrameTimeManager::getFrameCount() const noexcept -> uint64_t
{
    return _frameCounter;
}

auto FrameTimeManager::getMeanFrameRate() const noexcept -> float
{
    return 1.F / getMeanFrameTime();
}

auto FrameTimeManager::getMeanFrameTime() const noexcept -> float
{
    const auto currentTime = std::chrono::steady_clock::now();
    return std::chrono::duration<float>(currentTime - _start).count() / static_cast<float>(_frameCounter);
}

}
