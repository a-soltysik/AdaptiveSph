#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "object/Surface.h"
#include "panda/gfx/Camera.h"
#include "panda/gfx/Light.h"
#include "panda/gfx/vulkan/object/Object.h"

namespace panda::gfx::vulkan
{

class Scene
{
public:
    [[nodiscard]] auto getSize() const noexcept -> size_t;

    [[nodiscard]] auto getLights() const noexcept -> const Lights&;
    [[nodiscard]] auto getCamera() const noexcept -> const Camera&;

    [[nodiscard]] auto getCamera() noexcept -> Camera&;  // cppcheck-suppress functionConst

    auto setDomain(std::string name, const std::vector<Surface>& surfaces) -> Object&;
    [[nodiscard]] auto getDomain() const -> Object&;

    auto setParticleCount(uint32_t count) -> void;
    auto getParticleCount() const -> uint32_t;

    auto addObject(std::string name, const std::vector<Surface>& surfaces) -> Object&;
    auto removeObjectByName(std::string_view name) -> bool;
    [[nodiscard]] auto findObjectByName(std::string_view name) -> std::optional<Object*>;
    [[nodiscard]] auto findObjectByName(std::string_view name) const -> std::optional<const Object*>;
    [[nodiscard]] auto getObjects() const noexcept -> const std::vector<std::unique_ptr<Object>>&;

    [[nodiscard]] auto findLightByName(std::string_view name) -> std::variant<std::reference_wrapper<DirectionalLight>,
                                                                              std::reference_wrapper<PointLight>,
                                                                              std::reference_wrapper<SpotLight>,
                                                                              std::monostate>;
    [[nodiscard]] auto findLightByName(std::string_view name) const
        -> std::variant<std::reference_wrapper<const DirectionalLight>,
                        std::reference_wrapper<const PointLight>,
                        std::reference_wrapper<const SpotLight>,
                        std::monostate>;
    auto removeLightByName(std::string_view name) -> bool;

    [[nodiscard]] auto getAllNames() const noexcept -> const std::unordered_set<std::string_view>&;

    template <typename Light>
    auto addLight(std::string name) -> std::optional<std::reference_wrapper<Light>>
    {
        if constexpr (std::is_same_v<Light, PointLight>)
        {
            if (_lights.pointLights.size() >= maxLights)
            {
                return {};
            }
            _lights.pointLights.push_back({
                {.name = getUniqueName(std::move(name)), .ambient = {}, .diffuse = {}, .specular = {}, .intensity = {}},
                {},
                {}
            });
            _names.insert(_lights.pointLights.back().name);
            return _lights.pointLights.back();
        }
        else if constexpr (std::is_same_v<Light, DirectionalLight>)
        {
            if (_lights.directionalLights.size() >= maxLights)
            {
                return {};
            }
            _lights.directionalLights.push_back({
                {.name = getUniqueName(std::move(name)), .ambient = {}, .diffuse = {}, .specular = {}, .intensity = {}},
                {}
            });
            _names.insert(_lights.directionalLights.back().name);
            return _lights.directionalLights.back();
        }
        else if constexpr (std::is_same_v<Light, SpotLight>)
        {
            if (_lights.spotLights.size() >= maxLights)
            {
                return {};
            }
            _lights.spotLights.push_back({
                {{.name = getUniqueName(std::move(name)),
                  .ambient = {},
                  .diffuse = {},
                  .specular = {},
                  .intensity = {}},
                 {},
                 {}},
                {},
                {}
            });
            _names.insert(_lights.spotLights.back().name);
            return _lights.spotLights.back();
        }
        return {};
    }

private:
    auto getUniqueName(std::string name) -> std::string;
    auto removeName(std::string_view name) -> void;

    std::unique_ptr<Object> _domain;
    std::vector<std::unique_ptr<Object>> _objects;

    std::unordered_set<std::string_view> _names;
    Lights _lights;
    Camera _camera;
    uint32_t _particleCount = 0;
};

}
