#include <panda/Logger.h>

#include <cstddef>
#include <exception>
#include <span>

#include "App.hpp"

auto main(int argc, char** argv) -> int
{
    try
    {
        if (argc > 1)
        {
            return sph::App {
                std::span {argv, static_cast<size_t>(argc)}
                 [0]
            }
                .run();
        }

        return sph::App {}.run();
    }
    catch (const std::exception& e)
    {
        panda::log::Error("{}", e.what());
        return -1;
    }
    catch (...)
    {
        panda::log::Error("Unknown exception");
        return -1;
    }
}
