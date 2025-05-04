#pragma once

#include <cstddef>

namespace sph::cuda
{
template <typename T>
struct Span
{
    T* data;
    size_t size;
};
}
