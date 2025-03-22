#pragma once

#include <optional>

#include "Window.hpp"

namespace sph
{

class MovementHandler
{
public:
    struct Result
    {
        std::optional<float> x;
        std::optional<float> y;
        std::optional<float> z;
    };

    explicit MovementHandler(const Window& window);
    [[nodiscard]] auto getMovement() const -> Result;

private:
    const Window& _window;
};

}
