include("${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake")

set(CMAKE_CUDA_STANDARD 20)

add_library(AdaptiveSphCuda SHARED src/Kernel.cu src/SphSimulation.cu src/ImportedParticleMemory.cu)
target_compile_features(AdaptiveSphCuda PRIVATE cxx_std_20)
target_compile_definitions(AdaptiveSphCuda PRIVATE EXPORTING_CUDA_LIB)
target_include_directories(AdaptiveSphCuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(AdaptiveSphCuda PRIVATE sph_options sph_warnings)
sph_target_link_cuda(AdaptiveSphCuda)

if(WIN32)
    target_compile_definitions(AdaptiveSphCuda PRIVATE WIN32)
endif()

target_link_system_libraries(AdaptiveSphCuda PRIVATE glm)

set_target_properties(
    AdaptiveSphCuda
    PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
               RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
