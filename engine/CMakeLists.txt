file(GLOB_RECURSE SRC_LIST src/*.cpp)

add_library(AdaptiveSphEngine)

target_sources(AdaptiveSphEngine PRIVATE ${SRC_LIST})

target_include_directories(
    AdaptiveSphEngine
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/configured_files/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

find_package(Vulkan REQUIRED)
target_link_system_libraries(
    AdaptiveSphEngine
    PRIVATE
    AdaptiveSphCuda
    Vulkan::Vulkan
    glm
    fmt
    imgui
    ctre)

target_compile_definitions(
    AdaptiveSphEngine
    PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
           VULKAN_HPP_NO_EXCEPTIONS
           VULKAN_HPP_ASSERT_ON_RESULT=EXPECT
           VULKAN_HPP_ASSERT=DEBUG_EXPECT
           GLM_FORCE_RADIANS
           GLM_FORCE_DEPTH_ZERO_TO_ONE
           GLM_ENABLE_EXPERIMENTAL
           GLFW_INCLUDE_VULKAN)

if(WIN32)
    target_compile_definitions(AdaptiveSphEngine PUBLIC VK_USE_PLATFORM_WIN32_KHR)
else()
    target_compile_definitions(AdaptiveSphEngine PUBLIC VK_USE_PLATFORM_XLIB_KHR)
endif()

if(MSVC)
    target_compile_definitions(AdaptiveSphEngine PRIVATE NOMINMAX)
endif()

if(sph_ENABLE_PCH)
    target_precompile_headers(
        AdaptiveSphEngine
        PRIVATE
        <cstddef>
        <cstdint>
        <memory>
        <vector>
        <optional>
        <span>
        <string>
        <panda/utils/Assert.h>
        <vulkan/vulkan.hpp>
        <vulkan/vulkan_handles.hpp>
        <vulkan/vulkan_enums.hpp>
        <vulkan/vulkan_structs.hpp>)
endif()

add_subdirectory(shader)
add_subdirectory(config)
