#pragma once
#include <algorithm>
#include <chrono>
#include <functional>
#include <utility>
#include <vector>

namespace sph::utils
{

class TaskScheduler
{
public:
    void addTask(std::function<void()> callback, std::chrono::milliseconds period)
    {
        _tasks.emplace_back(Task {.callback = std::move(callback), .period = period});
    }

    void update(std::chrono::milliseconds deltaTime)
    {
        std::ranges::for_each(_tasks, [deltaTime](auto& task) {
            task.elapsed += deltaTime;
            if (task.elapsed >= task.period)
            {
                task.callback();
                task.elapsed = {};
            }
        });
    }

private:
    struct Task
    {
        std::function<void()> callback;
        std::chrono::milliseconds period;
        std::chrono::milliseconds elapsed {};
    };

    std::vector<Task> _tasks;
};
}
