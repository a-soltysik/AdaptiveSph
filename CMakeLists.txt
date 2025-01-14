cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

project(AdaptiveSph LANGUAGES CXX CUDA)

include(cmake/PreventInSourceBuilds.cmake)
include(ProjectOptions.cmake)

sph_setup_options()

sph_global_options()
include(Dependencies.cmake)
sph_setup_dependencies()

sph_local_options()

target_compile_features(sph_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

add_library(sph::sph_options ALIAS sph_options)
add_library(sph::sph_warnings ALIAS sph_warnings)

add_subdirectory(configured_files)

add_subdirectory(src)
add_subdirectory(cuda)

if(MSVC)
    get_all_installable_targets(all_targets)
    message("all_targets=${all_targets}")
    set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif()

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT AdaptiveSph)
