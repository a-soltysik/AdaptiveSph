#include "panda/gfx/vulkan/Scene.h"

#include <algorithm>
#include <ctre.hpp>  // NOLINT(misc-include-cleaner)
#include <ctre/wrapper.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "panda/Logger.h"
#include "panda/gfx/Camera.h"
#include "panda/gfx/Light.h"
#include "panda/gfx/vulkan/object/Object.h"
#include "panda/gfx/vulkan/object/Surface.h"
#include "panda/utils/Utils.h"

namespace panda::gfx::vulkan
{

auto Scene::getSize() const noexcept -> size_t
{
    return std::accumulate(_objects.begin(),
                           _objects.end(),
                           size_t {},
                           [](auto current, const auto& object) {
                               return current + object->getSurfaces().size();
                           }) +
           _lights.spotLights.size() + _lights.pointLights.size() + _lights.directionalLights.size();
}

auto Scene::addObject(std::string name, const std::vector<Surface>& surfaces) -> Object&
{
    auto newObject = std::make_unique<Object>(getUniqueName(std::move(name)));
    for (const auto& surface : surfaces)
    {
        newObject->addSurface(surface);
    }

    auto* objectPtr = newObject.get();
    _objects.push_back(std::move(newObject));

    _names.insert(objectPtr->getName());
    return *objectPtr;
}

auto Scene::getObjects() const noexcept -> const std::vector<std::unique_ptr<Object>>&
{
    return _objects;
}

auto Scene::getLights() const noexcept -> const Lights&
{
    return _lights;
}

auto Scene::getCamera() const noexcept -> const Camera&
{
    return _camera;
}

auto Scene::getCamera() noexcept -> Camera&
{
    return _camera;
}

auto Scene::removeObjectByName(std::string_view name) -> bool
{
    const auto objectIt = std::ranges::find(_objects, name, &Object::getName);
    if (objectIt != std::ranges::end(_objects))
    {
        removeName(name);
        _objects.erase(objectIt);
        return true;
    }
    return false;
}

auto Scene::findObjectByName(std::string_view name) -> std::optional<Object*>
{
    const auto objectIt = std::ranges::find(_objects, name, &Object::getName);
    if (objectIt != std::ranges::end(_objects))
    {
        return objectIt->get();
    }
    return {};
}

auto Scene::findObjectByName(std::string_view name) const -> std::optional<const Object*>
{
    const auto objectIt = std::ranges::find(_objects, name, &Object::getName);
    if (objectIt != std::ranges::end(_objects))
    {
        return objectIt->get();
    }
    return {};
}

auto Scene::findLightByName(std::string_view name) -> std::variant<std::reference_wrapper<DirectionalLight>,
                                                                   std::reference_wrapper<PointLight>,
                                                                   std::reference_wrapper<SpotLight>,
                                                                   std::monostate>
{
    const auto directionalLightIt = std::ranges::find(_lights.directionalLights, name, &BaseLight::name);
    if (directionalLightIt != std::ranges::end(_lights.directionalLights))
    {
        return std::ref(*directionalLightIt);
    }
    const auto pointLightIt = std::ranges::find(_lights.pointLights, name, &BaseLight::name);
    if (pointLightIt != std::ranges::end(_lights.pointLights))
    {
        return std::ref(*pointLightIt);
    }
    const auto spotLightIt = std::ranges::find(_lights.spotLights, name, &BaseLight::name);
    if (spotLightIt != std::ranges::end(_lights.spotLights))
    {
        return std::ref(*spotLightIt);
    }
    return std::monostate {};
}

auto Scene::findLightByName(std::string_view name) const -> std::variant<std::reference_wrapper<const DirectionalLight>,
                                                                         std::reference_wrapper<const PointLight>,
                                                                         std::reference_wrapper<const SpotLight>,
                                                                         std::monostate>
{
    const auto directionalLightIt = std::ranges::find(_lights.directionalLights, name, &BaseLight::name);
    if (directionalLightIt != std::ranges::end(_lights.directionalLights))
    {
        return std::cref(*directionalLightIt);
    }
    const auto pointLightIt = std::ranges::find(_lights.pointLights, name, &BaseLight::name);
    if (pointLightIt != std::ranges::end(_lights.pointLights))
    {
        return std::cref(*pointLightIt);
    }
    const auto spotLightIt = std::ranges::find(_lights.spotLights, name, &BaseLight::name);
    if (spotLightIt != std::ranges::end(_lights.spotLights))
    {
        return std::cref(*spotLightIt);
    }
    return std::monostate {};
}

auto Scene::removeLightByName(std::string_view name) -> bool
{
    const auto directionalLightIt = std::ranges::find(_lights.directionalLights, name, &BaseLight::name);
    if (directionalLightIt != std::ranges::end(_lights.directionalLights))
    {
        removeName(name);
        _lights.directionalLights.erase(directionalLightIt);
        return true;
    }
    const auto pointLightIt = std::ranges::find(_lights.pointLights, name, &BaseLight::name);
    if (pointLightIt != std::ranges::end(_lights.pointLights))
    {
        removeName(name);
        _lights.pointLights.erase(pointLightIt);
        return true;
    }
    const auto spotLightIt = std::ranges::find(_lights.spotLights, name, &BaseLight::name);
    if (spotLightIt != std::ranges::end(_lights.spotLights))
    {
        removeName(name);
        _lights.spotLights.erase(spotLightIt);
        return true;
    }
    return false;
}

auto Scene::getUniqueName(std::string name) -> std::string
{
    if (!_names.contains(name))
    {
        return std::move(name);
    }

    uint32_t maxNumber = 1;

    for (const auto currentName : _names)
    {
        if (auto match = ctre::match<R"(^.*#(\d+)$)">(currentName))
        {
            if (auto numberValue = utils::toNumber<uint32_t>(match.get<1>().to_view()); numberValue)
            {
                maxNumber = std::max(maxNumber, *numberValue + 1);
            }
        }
    }
    auto& prefix = name += '#';
    return prefix += utils::toString(maxNumber);
}

auto Scene::removeName(std::string_view name) -> void
{
    log::Info("Removed object with name {}", name);
    _names.erase(name);
}

auto Scene::getAllNames() const noexcept -> const std::unordered_set<std::string_view>&
{
    return _names;
}

auto Scene::setDomain(std::string name, const std::vector<Surface>& surfaces) -> Object&
{
    auto newObject = std::make_unique<Object>(getUniqueName(std::move(name)));
    for (const auto& surface : surfaces)
    {
        newObject->addSurface(surface);
    }

    if (_domain)
    {
        removeName(_domain->getName());
    }

    auto* objectPtr = newObject.get();
    _domain = std::move(newObject);

    _names.insert(objectPtr->getName());
    return *objectPtr;
}

auto Scene::getDomain() const -> Object&
{
    return *_domain;
}

auto Scene::setParticleCount(uint32_t count) -> void
{
    _particleCount = count;
}

auto Scene::getParticleCount() const -> uint32_t
{
    return _particleCount;
}

}
