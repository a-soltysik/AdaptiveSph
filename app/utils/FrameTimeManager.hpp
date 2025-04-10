#pragma once
#include <chrono>

namespace sph
{

class FrameTimeManager
{
public:
    auto update() -> void;

    [[nodiscard]] auto getDelta() const noexcept -> float;
    [[nodiscard]] auto getFrameCount() const noexcept -> uint64_t;
    [[nodiscard]] auto getMeanFrameRate() const noexcept -> float;
    [[nodiscard]] auto getMeanFrameTime() const noexcept -> float;

private:
    std::chrono::steady_clock::time_point _start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point _currentFrameStart = std::chrono::steady_clock::now();

    uint64_t _frameCounter {};
    float _deltaTime {};
};

}
