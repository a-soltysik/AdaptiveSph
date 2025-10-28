# from here:
#
# https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md

function(
        sph_set_project_warnings
        project_name
        WARNINGS_AS_ERRORS)
    set(MSVC_WARNINGS
            /W4
            /w14242
            /w14254
            /w14263
            /w14265
            /w14287
            /we4289
            /w14296
            /w14311
            /w14545
            /w14546
            /w14547
            /w14549
            /w14555
            /w14619
            /w14640
            /w14826
            /w14905
            /w14906
            /w14928
            /permissive-
    )

    set(CLANG_WARNINGS
            -Wall
            -Wextra
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
            -Wimplicit-fallthrough
            -Wswitch-enum)

    set(GCC_WARNINGS
            ${CLANG_WARNINGS}
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast
            -Wsuggest-override
    )

    set(CUDA_WARNINGS_GNU_CLANG
            -Wall
            -Wextra
            -Wshadow
            -Wnon-virtual-dtor
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
            -Wimplicit-fallthrough
    )

    if (WARNINGS_AS_ERRORS)
        message(TRACE "Warnings are treated as errors")
        list(APPEND CLANG_WARNINGS -Werror)
        list(APPEND GCC_WARNINGS -Werror)
        list(APPEND MSVC_WARNINGS /WX)
        list(APPEND CUDA_WARNINGS_GNU_CLANG -Werror)
    endif ()

    if (MSVC)
        set(PROJECT_WARNINGS ${MSVC_WARNINGS})
        set(CUDA_WARNINGS -Xcompiler=/W4)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        list(TRANSFORM CUDA_WARNINGS_GNU_CLANG PREPEND "-Xcompiler=")
        set(PROJECT_WARNINGS ${CLANG_WARNINGS})
        set(CUDA_WARNINGS ${CUDA_WARNINGS_GNU_CLANG})
        set(LINKER_WARNINGS -Wl,--no-undefined -Wl,--no-allow-shlib-undefined)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(TRANSFORM CUDA_WARNINGS_GNU_CLANG PREPEND "-Xcompiler=")
        set(PROJECT_WARNINGS ${GCC_WARNINGS})
        set(CUDA_WARNINGS ${CUDA_WARNINGS_GNU_CLANG})
        set(LINKER_WARNINGS -Wl,--no-undefined -Wl,--no-allow-shlib-undefined)
    else ()
        message(AUTHOR_WARNING "No compiler warnings set for CXX compiler: '${CMAKE_CXX_COMPILER_ID}'")
        return()
    endif ()

    target_compile_options(${project_name} INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS}>
            $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_WARNINGS}>
    )
    target_link_options(${project_name} INTERFACE
            $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:${LINKER_WARNINGS}>)
endfunction()
