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
    [[nodiscard]] auto getXMovement() const -> std::optional<float>;
    [[nodiscard]] auto getYMovement() const -> std::optional<float>;
    [[nodiscard]] auto getZMovement() const -> std::optional<float>;

    const Window& _window;
};

}
