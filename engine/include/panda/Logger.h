#pragma once

#include <fmt/base.h>
#include <fmt/core.h>
#include <fmt/os.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "panda/Common.h"

namespace panda::log
{

enum class Level : uint8_t
{
    Debug,
    Info,
    Warning,
    Error
};

template <typename... Args>
struct Debug
{
    explicit Debug(std::string_view format,
                   Args&&... args,
                   std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Info
{
    explicit Info(std::string_view format,
                  Args&&... args,
                  std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Warning
{
    explicit Warning(std::string_view format,
                     Args&&... args,
                     std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Error
{
    explicit Error(std::string_view format,
                   Args&&... args,
                   std::source_location location = std::source_location::current()) noexcept;
};

namespace internal
{
struct LogDispatcher;
}

class FileLogger
{
public:
    struct LogData
    {
        uint32_t index;
        std::string message;
        std::source_location location;
        std::chrono::system_clock::time_point time;
        Level level;
    };

    PD_DELETE_ALL(FileLogger);
    ~FileLogger();

    auto setLevels(std::span<const Level> newLevels) -> void;
    auto start() -> void;
    auto stop() -> void;
    auto terminate() -> void;
    auto setBufferSize(size_t size) -> void;
    [[nodiscard]] auto getBuffer() const -> const std::vector<LogData>&;

    static auto instance() -> FileLogger&;

private:
    friend struct internal::LogDispatcher;

    FileLogger() = default;
    auto log(Level level, std::string_view message, const std::source_location& location) -> void;
    auto flush() -> void;

    std::set<Level> _levels = {Level::Debug, Level::Info, Level::Warning, Level::Error};
    std::vector<LogData> _buffer;
    std::unique_ptr<fmt::ostream> _file;
    size_t _bufferSize = 1000;
    uint32_t _currentIndex = 0;
    bool _isStarted = false;
};

namespace internal
{

struct LogDispatcher
{
    static auto log(Level level, std::string_view message, const std::source_location& location) noexcept -> void;
};

}

template <typename... Args>
Debug<Args...>::Debug(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Debug,
                                 fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                 location);
}

template <typename... Args>
Info<Args...>::Info(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Info, fmt::format(fmt::runtime(format), std::forward<Args>(args)...), location);
}

template <typename... Args>
Warning<Args...>::Warning(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Warning,
                                 fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                 location);
}

template <typename... Args>
Error<Args...>::Error(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Error,
                                 fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                 location);
}

template <typename... Args>
Debug(std::string_view, Args&&...) -> Debug<Args...>;

template <typename... Args>
Info(std::string_view, Args&&...) -> Info<Args...>;

template <typename... Args>
Warning(std::string_view, Args&&...) -> Warning<Args...>;

template <typename... Args>
Error(std::string_view, Args&&...) -> Error<Args...>;

}