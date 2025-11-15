#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace sph::utils
{

class Task
{
public:
    virtual ~Task() = default;
    virtual void update(std::chrono::milliseconds deltaTime, uint32_t deltaFrame) = 0;
};

class PerFrameTask final : public Task
{
public:
    PerFrameTask(std::function<void()> task, uint32_t period)
        : _task {std::move(task)},
          _period {period}
    {
    }

    void update([[maybe_unused]] std::chrono::milliseconds deltaTime, uint32_t deltaFrame) override
    {
        _frameCounter += deltaFrame;
        if (_frameCounter >= _period)
        {
            _task();
            _frameCounter = {};
        }
    }

private:
    const std::function<void()> _task;
    const uint32_t _period;
    uint32_t _frameCounter {};
};

class PerTimeTask final : public Task
{
public:
    PerTimeTask(std::function<void()> task, std::chrono::milliseconds period)
        : _task {std::move(task)},
          _period {period}
    {
    }

    void update(std::chrono::milliseconds deltaTime, [[maybe_unused]] uint32_t deltaFrame) override
    {
        _timeCounter += deltaTime;
        if (_timeCounter >= _period)
        {
            _task();
            _timeCounter = {};
        }
    }

private:
    std::function<void()> _task;
    std::chrono::milliseconds _period;
    std::chrono::milliseconds _timeCounter {};
};

class TaskScheduler
{
public:
    void addTask(std::unique_ptr<Task> task)
    {
        _tasks.emplace_back(std::move(task));
    }

    void update(std::chrono::milliseconds deltaTime, uint32_t deltaFrame)
    {
        std::ranges::for_each(_tasks, [deltaTime, deltaFrame](const auto& task) {
            task->update(deltaTime, deltaFrame);
        });
    }

private:
    std::vector<std::unique_ptr<Task>> _tasks;
};
}
