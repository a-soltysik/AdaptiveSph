macro(sph_target_link_cuda target_name)
    set_target_properties(${target_name} PROPERTIES
            CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_RUNTIME_LIBRARY Shared
            POSITION_INDEPENDENT_CODE ON
            CUDA_SEPARABLE_COMPILATION OFF
    )

    if (ASPH_CUDA_ENABLE_CUSTOM_ARCHITECTURES)
        set_target_properties(${target_name} PROPERTIES
                CUDA_ARCHITECTURES "${ASPH_CUDA_ARCHITECTURES}"
        )
    endif ()

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-g>
        )
        if (ASPH_CUDA_ENABLE_DEBUG)
            target_compile_options(${target_name} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-G>
            )
        endif ()
    else ()
        if (ASPH_CUDA_ENABLE_LINEINFO)
            target_compile_options(${target_name} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            )
        endif ()
        target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-O3>
        )
    endif ()

    target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012>
    )
endmacro()
