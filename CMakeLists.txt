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

set(ASPH_CONFIG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/configs")
file(MAKE_DIRECTORY "${ASPH_CONFIG_OUTPUT_DIR}")
configure_file(configs/base.json "${ASPH_CONFIG_OUTPUT_DIR}/base.json" COPYONLY)
configure_file(configs/adaptive.json "${ASPH_CONFIG_OUTPUT_DIR}/adaptive.json" COPYONLY)
configure_file(configs/poiseuille.json "${ASPH_CONFIG_OUTPUT_DIR}/poiseuille.json" COPYONLY)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


sph_setup_options()
sph_global_options()

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

message(STATUS "AdaptiveSph configuration: CMake ${CMAKE_VERSION}, ${CMAKE_BUILD_TYPE}")
message(STATUS "  Host compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER} (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
