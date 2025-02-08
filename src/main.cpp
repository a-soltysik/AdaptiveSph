#include <panda/Logger.h>

#include <exception>

#include "App.hpp"

auto main() -> int
{
    try
    {
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
