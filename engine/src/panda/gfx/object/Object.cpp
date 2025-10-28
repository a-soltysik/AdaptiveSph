#include "panda/gfx/object/Object.h"

#include <string>
#include <utility>
#include <vector>

#include "panda/gfx/object/Surface.h"

namespace panda::gfx
{

auto Object::getId() const noexcept -> Id
{
    return _id;
}

Object::Object(std::string name)
    : _name {std::move(name)},
      _id {currentId++}
{
}

auto Object::getName() const noexcept -> const std::string&
{
    return _name;
}

auto Object::addSurface(const Surface& surface) -> void
{
    surfaces.push_back(surface);
}

auto Object::getSurfaces() const noexcept -> const std::vector<Surface>&
{
    return surfaces;
}

auto Object::getNextId() -> Id
{
    return currentId;
}
}
