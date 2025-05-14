#include "panda/Logger.h"

#include <fmt/chrono.h>  // NOLINT(misc-include-cleaner)
#include <fmt/format.h>
#include <fmt/os.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace panda::log
{

namespace
{

[[nodiscard]] constexpr auto getLevelTag(Level level) -> std::string_view
{
    using namespace std::string_view_literals;
    switch (level)
    {
    case Level::Debug:
        return "[DBG]"sv;
    case Level::Info:
        return "[INF]"sv;
    case Level::Warning:
        return "[WRN]"sv;
    case Level::Error:
        return "[ERR]"sv;
    default:
        [[unlikely]] return "[???]"sv;
    }
}

[[nodiscard]] constexpr auto getFunctionName(std::string_view function) -> std::string_view
{
    const auto nameWithReturnType = function.substr(0, function.find('('));
    return nameWithReturnType.substr(nameWithReturnType.rfind(' ') + 1);
}

}

namespace internal
{

auto LogDispatcher::log([[maybe_unused]] Level level,
                        [[maybe_unused]] std::string_view message,
                        [[maybe_unused]] const std::source_location& location) noexcept -> void
{
    try
    {
        FileLogger::instance().log(level, message, location);
    }
    catch (...)  //NOLINT(bugprone-empty-catch)
    {
    }
}

}

auto FileLogger::instance() -> FileLogger&
{
    static FileLogger config;
    return config;
}

auto FileLogger::log(Level level, std::string_view message, const std::source_location& location) -> void
{
    if (!_isStarted)
    {
        return;
    }
    if (_levels.contains(level))
    {
        _buffer.push_back(LogData {.index = _currentIndex++,
                                   .message = std::string {message},
                                   .location = location,
                                   .time = std::chrono::system_clock::now(),
                                   .level = level});
        if (_buffer.size() >= _bufferSize)
        {
            flush();
        }
        fmt::println("{:%H:%M:%S} {} {}:{}, {}",
                     std::chrono::system_clock::now(),
                     getLevelTag(level),
                     getFunctionName(location.function_name()),
                     location.line(),
                     message);
    }
}

auto FileLogger::setLevels(std::span<const Level> newLevels) -> void
{
    _levels = {newLevels.begin(), newLevels.end()};
}

auto FileLogger::start() -> void
{
    if (_file)
    {
        _isStarted = true;
        return;
    }

    const auto time = std::chrono::system_clock::now();
    static constexpr auto logsDir = std::string_view {"logs"};

    try
    {
        std::filesystem::create_directory(logsDir);
        const auto filename =
            fmt::format("{}/{:%F_%H_%M_%S.pdlog}",
                        logsDir,
                        std::chrono::floor<std::chrono::seconds>(time));  //NOLINT(misc-include-cleaner)
        _file = std::make_unique<fmt::ostream>(fmt::output_file(filename));
        _isStarted = true;

        Info("File logger started");
    }
    catch (const std::system_error& e)
    {
        _isStarted = false;
        Error(e.what());
    }
}

auto FileLogger::stop() -> void
{
    _isStarted = false;
    flush();
}

auto FileLogger::setBufferSize(size_t size) -> void
{
    _bufferSize = size;
}

auto FileLogger::flush() -> void
{
    for (const auto& line : _buffer)
    {
        _file->print("{:%H:%M:%S} {} {}:{}, {}\n",
                     line.time,
                     getLevelTag(line.level),
                     getFunctionName(line.location.function_name()),
                     line.location.line(),
                     line.message);
    }
    _buffer.clear();
}

FileLogger::~FileLogger()
{
    flush();
}

auto FileLogger::terminate() -> void
{
    flush();
    _file->close();
}

auto FileLogger::getBuffer() const -> const std::vector<LogData>&
{
    return _buffer;
}

}
