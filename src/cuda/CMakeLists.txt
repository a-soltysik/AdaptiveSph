include("${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake")

set(CMAKE_CUDA_STANDARD 20)
add_library(AdaptiveSphCuda STATIC main.cu)

target_compile_features(AdaptiveSphCuda PRIVATE cxx_std_20)
target_link_libraries(AdaptiveSphCuda PRIVATE sph_options sph_warnings)
sph_target_link_cuda(AdaptiveSphCuda)
