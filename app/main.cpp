#include <panda/Logger.h>
#include <panda/utils/Assert.h>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "App.hpp"

auto main(int argc, char** argv) -> int
{
    try
    {
        const auto arguments = std::span {argv, static_cast<std::size_t>(argc)};
        if (argc == 2 &&
            (std::string_view {arguments.back()} == "--help" || std::string_view {arguments.back()} == "-h"))
        {
            panda::log::FileLogger::instance().start();
            panda::log::Info(
                "Usage: AdaptiveSph [config.json]\n"
                "Without an argument, loads configs/base.json beside the executable.");
            return 0;
        }

        panda::expect(argc >= 1 && argc <= 2, "Expected at most one configuration path. Use --help for usage.");
        const auto executableDirectory = std::filesystem::absolute(arguments.front()).parent_path();
        const auto configPath = argc == 2 ? std::filesystem::absolute(arguments.back()).string()
                                          : (executableDirectory / "configs" / "base.json").string();
        std::filesystem::current_path(executableDirectory);
        return sph::App {configPath}.run();
    }
    catch (const std::exception& error)
    {
        panda::log::FileLogger::instance().start();
        panda::log::Error("Unhandled exception: {}", error.what());
        return 1;
    }
    catch (...)
    {
        panda::log::FileLogger::instance().start();
        panda::log::Error("Unhandled exception of unknown type");
        return 1;
    }
}
