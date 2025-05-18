include("${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake")

set(CMAKE_CUDA_STANDARD 20)

add_library(
        AdaptiveSphCuda SHARED
        src/Algorithm.cu
        src/SphSimulation.cu
        src/ImportedParticleMemory.cu
        src/device/Kernel.cu
        src/refinement/ParticleOperations.cu

        src/AdaptiveSphSimulation.cu
        src/SimulationFactory.cu
        src/common/Utils.cu
        src/refinement/criteria/VelocityCriterion.cu
        src/refinement/criteria/VorticityCriterion.cu
        src/refinement/criteria/CurvatureCriterion.cu
        src/refinement/criteria/InterfaceCriterion.cu
        src/common/Iteration.cu)
target_compile_features(AdaptiveSphCuda PRIVATE cxx_std_20)
target_compile_definitions(AdaptiveSphCuda PRIVATE EXPORTING_CUDA_LIB)
target_include_directories(AdaptiveSphCuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(AdaptiveSphCuda PRIVATE sph_options sph_warnings)
sph_target_link_cuda(AdaptiveSphCuda)

if (WIN32)
    target_compile_definitions(AdaptiveSphCuda PRIVATE WIN32)
endif ()

target_link_system_libraries(AdaptiveSphCuda PRIVATE glm)
set_property(TARGET AdaptiveSphCuda PROPERTY CUDA_ARCHITECTURES 89)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(AdaptiveSphCuda
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-O3,--allow-expensive-optimizations=true>)
else ()
    target_compile_options(AdaptiveSphCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G> $<$<COMPILE_LANGUAGE:CUDA>:-g>)
endif ()
target_compile_definitions(AdaptiveSphCuda PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_options(AdaptiveSphCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set_target_properties(
        AdaptiveSphCuda
        PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
