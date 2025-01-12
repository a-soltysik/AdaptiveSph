add_executable(AdaptiveSph main.cpp)

target_link_libraries(AdaptiveSph PRIVATE sph::sph_options sph::sph_warnings AdaptiveSphCuda)

target_link_system_libraries(
    AdaptiveSph
    PRIVATE
    PD::Engine
    fmt::fmt)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER
        ${CMAKE_CXX_COMPILER}
        CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
target_include_directories(AdaptiveSph PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

set_target_properties(AdaptiveSph PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
