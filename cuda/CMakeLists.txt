include("${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake")

add_library(
        AdaptiveSphCuda SHARED
        src/algorithm/Algorithm.cu
        src/simulation/SphSimulation.cu
        src/memory/ImportedParticleMemory.cu
        src/algorithm/kernels/Kernel.cu
        src/algorithm/adaptive/AdaptiveAlgorithm.cu

        src/simulation/adaptive/AdaptiveSphSimulation.cu
        src/simulation/SimulationFactory.cu
        src/utils/Utils.cu
        src/simulation/adaptive/refinement/criteria/VelocityCriterion.cu
        src/simulation/adaptive/refinement/criteria/VorticityCriterion.cu
        src/simulation/adaptive/refinement/criteria/CurvatureCriterion.cu
        src/simulation/adaptive/refinement/criteria/InterfaceCriterion.cu
        src/utils/Iteration.cu)

add_library(AdaptiveSph::Cuda ALIAS AdaptiveSphCuda)

target_include_directories(AdaptiveSphCuda
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AdaptiveSphCuda PRIVATE sph_options sph_warnings)
target_link_system_libraries(AdaptiveSphCuda PRIVATE glm)
sph_target_link_cuda(AdaptiveSphCuda)

target_compile_definitions(AdaptiveSphCuda
        PRIVATE
        ASPH_CUDA_EXPORTS
        GLM_ENABLE_EXPERIMENTAL
        PUBLIC
        $<$<PLATFORM_ID:Windows>:WIN32>
)

set_target_properties(
        AdaptiveSphCuda
        PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set_target_properties(AdaptiveSphCuda PROPERTIES FOLDER "Libraries")
