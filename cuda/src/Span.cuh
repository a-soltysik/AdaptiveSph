#pragma once

#include <cstdint>

namespace sph::cuda
{
template <typename T>
struct Span
{
    T* data;
    size_t size;
};
}
