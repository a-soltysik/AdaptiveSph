add_executable(AdaptiveSph)

target_sources(
    AdaptiveSph
    PRIVATE main.cpp
            Window.cpp
            mesh/UvSphere.cpp
            mesh/InvertedCube.cpp
            input_handler/KeyboardHandler.cpp
            input_handler/MouseHandler.cpp
            movement_handler/MovementHandler.cpp
            movement_handler/RotationHandler.cpp
            gui/SimulationDataGui.cpp
            utils/FrameTimeManager.cpp
            App.cpp)

target_link_libraries(
    AdaptiveSph
    PRIVATE sph::sph_options
            sph::sph_warnings
            AdaptiveSphCuda
            AdaptiveSphEngine)

find_package(Vulkan REQUIRED)
target_link_system_libraries(
    AdaptiveSph
    PRIVATE
    Vulkan::Vulkan
    fmt::fmt
    glfw
    glm
    imgui)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER
        ${CMAKE_CXX_COMPILER}
        CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)

target_include_directories(AdaptiveSph PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(AdaptiveSph PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

if(MSVC)
    target_compile_definitions(AdaptiveSph PRIVATE NOMINMAX)
endif()

set_target_properties(
    AdaptiveSph
    PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
               RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

target_compile_definitions(AdaptiveSph PRIVATE GLFW_INCLUDE_VULKAN GLM_ENABLE_EXPERIMENTAL)

if(sph_ENABLE_PCH)
    target_precompile_headers(
        ${APP_TARGET_NAME}
        PRIVATE
        <string>
        <vector>
        <GLFW/glfw3.h>
        <panda/Logger.h>)
endif()
