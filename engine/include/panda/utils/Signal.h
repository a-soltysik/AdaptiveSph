#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace panda::utils
{

template <typename... Args>
class Sender;
template <typename... Args>
class Receiver;

template <typename... Args>
class Signal
{
public:
    using SenderT = Sender<Args...>;
    using ReceiverT = Receiver<Args...>;
    using ChannelT = std::function<void(Args...)>;

    auto registerSender() const -> SenderT;
    auto connect(ChannelT&& connection) -> ReceiverT;
    auto disconnect(const ReceiverT& receiver) -> void;

private:
    friend class Sender<Args...>;

    template <std::convertible_to<Args>... Params>
    auto emit(const Params&... params) const -> void
    {
        std::lock_guard<std::mutex> lock {_connectionsMutex};
        for ([[maybe_unused]] auto& [key, value] : _connections)
        {
            value(params...);
        }
    }

    std::vector<std::pair<size_t, ChannelT>> _connections;
    mutable std::mutex _connectionsMutex;
};

template <typename... Args>
class Receiver
{
public:
    Receiver() = default;
    Receiver(const Receiver&) = delete;

    Receiver(Receiver&& rhs) noexcept
        : _signal {rhs._signal},
          _id {rhs._id}
    {
        rhs._signal = nullptr;
    }

    auto operator=(const Receiver&) -> Receiver& = delete;

    auto operator=(Receiver&& rhs) noexcept -> Receiver&
    {
        if (_signal)
        {
            _signal->disconnect(*this);
        }
        _signal = rhs._signal;
        _id = rhs._id;
        rhs._signal = nullptr;
        return *this;
    }

    ~Receiver() noexcept
    {
        if (_signal == nullptr)
        {
            return;
        }
        _signal->disconnect(*this);
    }

    [[nodiscard]] auto getId() const noexcept -> size_t
    {
        return _id;
    }

private:
    friend class Signal<Args...>;

    explicit Receiver(Signal<Args...>& signal)
        : _signal {&signal},
          _id {currentId++}
    {
    }

    inline static size_t currentId = 1;

    Signal<Args...>* _signal {};
    size_t _id {};
};

template <typename... Args>
class Sender
{
public:
    explicit Sender(const Signal<Args...>& signal)
        : _signal {signal}
    {
    }

    Sender(const Sender&) = delete;
    Sender(Sender&&) noexcept = default;
    auto operator=(const Sender&) -> Sender& = delete;
    auto operator=(Sender&&) noexcept -> Sender& = default;
    ~Sender() = default;

    template <std::convertible_to<Args>... Params>
    auto operator()(const Params&... params) const -> void
    {
        _signal.emit(params...);
    }

private:
    const Signal<Args...>& _signal;
};

template <typename... Args>
[[nodiscard]] auto Signal<Args...>::registerSender() const -> SenderT
{
    return SenderT {*this};
}

template <typename... Args>
[[nodiscard]] auto Signal<Args...>::connect(ChannelT&& connection) -> ReceiverT
{
    std::lock_guard<std::mutex> lock {_connectionsMutex};
    auto receiver = ReceiverT {*this};
    _connections.emplace_back(receiver.getId(), std::move(connection));
    return receiver;
}

template <typename... Args>
auto Signal<Args...>::disconnect(const ReceiverT& receiver) -> void
{
    std::lock_guard<std::mutex> lock {_connectionsMutex};
    std::erase_if(_connections, [&receiver](const auto& elem) noexcept {
        return elem.first == receiver.getId();
    });
}

}
