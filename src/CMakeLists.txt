add_executable(AdaptiveSph main.cpp)
#sph_suppress_ipo()
add_subdirectory(cuda)
#sph_resume_ipo()

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
