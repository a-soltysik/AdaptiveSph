file(GLOB_RECURSE ENGINE_SRC_LIST src/*.cpp)

add_library(AdaptiveSphEngine STATIC ${ENGINE_SRC_LIST})
add_library(AdaptiveSph::Engine ALIAS AdaptiveSphEngine)

target_include_directories(
        AdaptiveSphEngine
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/configured_files/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


find_package(Vulkan REQUIRED)
target_link_system_libraries(
        AdaptiveSphEngine
        PRIVATE
        Vulkan::Vulkan
        glm::glm
        fmt::fmt
        imgui::imgui
        ctre::ctre)

target_link_libraries(
        AdaptiveSphEngine
        PUBLIC
        sph::sph_options
        sph::sph_warnings
        AdaptiveSph::Cuda
)

target_compile_definitions(
        AdaptiveSphEngine
        PUBLIC
        VULKAN_HPP_ASSERT_ON_RESULT=EXPECT
        VULKAN_HPP_ASSERT=DEBUG_EXPECT
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_EXCEPTIONS=1
        VULKAN_HPP_NO_CONSTRUCTORS=1
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
        GLFW_INCLUDE_VULKAN
        $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
        $<$<PLATFORM_ID:Linux>:VK_USE_PLATFORM_XLIB_KHR>
        PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
)

if (SPH_ENABLE_PCH)
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
endif ()

add_subdirectory(config)
add_subdirectory(shader)
add_dependencies(AdaptiveSphEngine AdaptiveSphShader)

set_target_properties(AdaptiveSphEngine PROPERTIES FOLDER "Libraries")
