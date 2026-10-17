add_executable(AdaptiveSph)

target_sources(
        AdaptiveSph
        PRIVATE main.cpp
        ui/Window.cpp
        mesh/InvertedCube.cpp
        input/KeyboardHandler.cpp
        input/MouseHandler.cpp
        movement/MovementHandler.cpp
        movement/RotationHandler.cpp
        ui/SimulationDataGui.cpp
        utils/FrameTimeManager.cpp
        utils/Configuration.cpp
        App.cpp)

target_link_libraries(
        AdaptiveSph
        PRIVATE
        sph::sph_options
        sph::sph_warnings
        AdaptiveSph::Engine)

find_package(Vulkan REQUIRED)
target_link_system_libraries(
        AdaptiveSph
        PRIVATE
        Vulkan::Vulkan
        fmt::fmt
        glfw
        glm
        imgui
        nlohmann_json)

target_include_directories(AdaptiveSph PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(AdaptiveSph PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

set_target_properties(
        AdaptiveSph
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

target_compile_definitions(AdaptiveSph
        PRIVATE
        GLFW_INCLUDE_VULKAN
        GLM_ENABLE_EXPERIMENTAL
        $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
)

if (sph_ENABLE_PCH)
    target_precompile_headers(
            ${APP_TARGET_NAME}
            PRIVATE
            <string>
            <vector>
            <GLFW/glfw3.h>
            <panda/Logger.h>)
endif ()
