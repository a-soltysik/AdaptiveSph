cmake_minimum_required(VERSION 3.25)

set(CMAKE_OBJECT_PATH_MAX 500)

include(cmake/Setup.cmake)
include(ProjectOptions.cmake)
include(Dependencies.cmake)

sph_assure_out_of_source_builds()

project(
        AdaptiveSph
        DESCRIPTION "Adaptive SPH simulation with real-time Vulkan rendering"
        LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


sph_setup_options()
sph_global_options()

if (SPH_ENABLE_COMPILE_COMMANDS)
    sph_enable_compile_commands()
endif ()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


sph_setup_dependencies()

sph_local_options()

target_compile_features(sph_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

add_library(sph::sph_options ALIAS sph_options)
add_library(sph::sph_warnings ALIAS sph_warnings)

add_subdirectory(configured_files)

add_subdirectory(app)
add_subdirectory(engine)
add_subdirectory(cuda)

if (MSVC)
    get_all_installable_targets(all_targets)
    message(" all_targets=${all_targets}")
    set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif ()

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT AdaptiveSph)

message(STATUS "╔════════════════════════════════════════════════════╗")
message(STATUS "║ Adaptive SPH Configuration Summary                 ║")
message(STATUS "╠════════════════════════════════════════════════════╣")
message(STATUS " CMake version: ${CMAKE_VERSION}")
message(STATUS " Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS " CUDA standard: C++${CMAKE_CUDA_STANDARD}")
message(STATUS " C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS " CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
message(STATUS " Features:")
message(STATUS " IPO/LTO: ${ASPH_ENABLE_IPO}")
message(STATUS " PCH: ${ASPH_ENABLE_PCH}")
message(STATUS " Unity build: ${ASPH_ENABLE_UNITY_BUILD}")
message(STATUS " Fast math: ${ASPH_ENABLE_FAST_MATH}")
message(STATUS " ccache: ${ASPH_ENABLE_CACHE}")
message(STATUS "")
message(STATUS " Quality:")
message(STATUS " Warnings: ${ASPH_ENABLE_WARNINGS}")
message(STATUS " Warnings as errors: ${ASPH_ENABLE_WARNINGS_AS_ERRORS}")
message(STATUS " clang-tidy: ${ASPH_ENABLE_CLANG_TIDY}")
message(STATUS " cppcheck: ${ASPH_ENABLE_CPPCHECK}")
message(STATUS "")
message(STATUS " Sanitizers:")
message(STATUS " Address: ${ASPH_ENABLE_SANITIZER_ADDRESS}")
message(STATUS " UB: ${ASPH_ENABLE_SANITIZER_UNDEFINED}")
message(STATUS " Thread: ${ASPH_ENABLE_SANITIZER_THREAD}")
message(STATUS " Memory: ${ASPH_ENABLE_SANITIZER_MEMORY}")
message(STATUS " Leak: ${ASPH_ENABLE_SANITIZER_LEAK}")
message(STATUS "")
message(STATUS " Security:")
message(STATUS " Hardening: ${ASPH_ENABLE_HARDENING}")
message(STATUS " Global hardening: ${ASPH_ENABLE_GLOBAL_HARDENING}")
message(STATUS "╚════════════════════════════════════════════════════╝")