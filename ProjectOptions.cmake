include(cmake/SystemLink.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

macro(sph_setup_options)
    option(ASPH_ENABLE_HARDENING "Enable hardening" OFF)
    cmake_dependent_option(
            ASPH_ENABLE_GLOBAL_HARDENING
            "Attempt to push hardening options to built dependencies"
            ON
            ASPH_ENABLE_HARDENING
            OFF)

    option(ASPH_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(ASPH_ENABLE_WARNINGS "Enable warnings" OFF)
    option(ASPH_ENABLE_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(ASPH_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(ASPH_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(ASPH_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(ASPH_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(ASPH_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(ASPH_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(ASPH_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(ASPH_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(ASPH_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(ASPH_ENABLE_IWYU "Enable include-what-you-use analysis" OFF)
    option(ASPH_ENABLE_PCH "Enable precompiled headers" OFF)
    option(ASPH_ENABLE_CACHE "Enable ccache" OFF)
    option(ASPH_ENABLE_COMPILE_COMMANDS "Enable support for compile_commnads.json" OFF)
    option(ASPH_ENABLE_FAST_MATH "Enable fast math compilation flags" OFF)
    option(ASPH_CUDA_ENABLE_CUSTOM_ARCHITECTURES "Enable choosing CUDA architectures" OFF)
    option(ASPH_CUDA_ENABLE_DEBUG "Enable CUDA debug symbols (-G)" OFF)
    option(ASPH_CUDA_ENABLE_LINEINFO "Enable CUDA line info" OFF)


    if (NOT PROJECT_IS_TOP_LEVEL)
        mark_as_advanced(
                ASPH_ENABLE_IPO
                ASPH_ENABLE_WARNINGS
                ASPH_ENABLE_WARNINGS_AS_ERRORS
                ASPH_ENABLE_USER_LINKER
                ASPH_ENABLE_SANITIZER_ADDRESS
                ASPH_ENABLE_SANITIZER_LEAK
                ASPH_ENABLE_SANITIZER_UNDEFINED
                ASPH_ENABLE_SANITIZER_THREAD
                ASPH_ENABLE_SANITIZER_MEMORY
                ASPH_ENABLE_UNITY_BUILD
                ASPH_ENABLE_CLANG_TIDY
                ASPH_ENABLE_CPPCHECK
                ASPH_ENABLE_IWYU
                ASPH_ENABLE_PCH
                ASPH_ENABLE_CACHE
                ASPH_ENABLE_COMPILE_COMMANDS
                ASPH_ENABLE_FAST_MATH
                ASPH_CUDA_ENABLE_CUSTOM_ARCHITECTURES OFF
                ASPH_CUDA_ENABLE_DEBUG OFF
                ASPH_CUDA_ENABLE_LINEINFO OFF)
    endif ()

endmacro()

macro(sph_global_options)
    if (ASPH_ENABLE_IPO)
        include(cmake/InterproceduralOptimization.cmake)
        sph_enable_ipo()
    endif ()

    if (ASPH_ENABLE_HARDENING AND ASPH_ENABLE_GLOBAL_HARDENING)
        include(cmake/Hardening.cmake)
        if (ASPH_ENABLE_SANITIZER_UNDEFINED
                OR ASPH_ENABLE_SANITIZER_ADDRESS
                OR ASPH_ENABLE_SANITIZER_THREAD
                OR ASPH_ENABLE_SANITIZER_LEAK)
            set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
        else ()
            set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
        endif ()
        sph_enable_hardening(sph_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
    endif ()
endmacro()

macro(sph_local_options)
    if (PROJECT_IS_TOP_LEVEL)
        include(cmake/StandardProjectSettings.cmake)
    endif ()

    if (ASPH_ENABLE_COMPILE_COMMANDS)
        include(cmake/CompileCommands.cmake)
        sph_enable_compile_commands()
    endif ()

    add_library(sph_warnings INTERFACE)
    add_library(sph_options INTERFACE)

    if (ASPH_ENABLE_WARNINGS)
        include(cmake/CompilerWarnings.cmake)
        sph_set_project_warnings(
                sph_warnings
                ${ASPH_ENABLE_WARNINGS_AS_ERRORS})
    endif ()

    if (ASPH_ENABLE_USER_LINKER)
        include(cmake/Linker.cmake)
        sph_configure_linker(sph_options)
    endif ()

    include(cmake/Sanitizers.cmake)
    sph_enable_sanitizers(
            sph_options
            ${ASPH_ENABLE_SANITIZER_ADDRESS}
            ${ASPH_ENABLE_SANITIZER_LEAK}
            ${ASPH_ENABLE_SANITIZER_UNDEFINED}
            ${ASPH_ENABLE_SANITIZER_THREAD}
            ${ASPH_ENABLE_SANITIZER_MEMORY})

    set_target_properties(sph_options PROPERTIES UNITY_BUILD ${ASPH_ENABLE_UNITY_BUILD})

    if (ASPH_ENABLE_PCH)
        target_precompile_headers(
                sph_options
                INTERFACE
                <vector>
                <string>
                <utility>)
    endif ()

    if (ASPH_ENABLE_CACHE)
        include(cmake/Cache.cmake)
        sph_enable_cache()
    endif ()

    include(cmake/StaticAnalyzers.cmake)
    if (ASPH_ENABLE_CLANG_TIDY)
        sph_enable_clang_tidy(sph_options ${ASPH_ENABLE_WARNINGS_AS_ERRORS})
    endif ()

    if (ASPH_ENABLE_CPPCHECK)
        sph_enable_cppcheck(${ASPH_ENABLE_WARNINGS_AS_ERRORS})
    endif ()

    if (ASPH_ENABLE_IWYU)
        sph_enable_include_what_you_use()
    endif ()

    if (ASPH_ENABLE_FAST_MATH)
        include(cmake/Optimizations.cmake)
        sph_enable_fast_math(sph_options)
    endif ()

    if (ASPH_ENABLE_HARDENING AND NOT ASPH_ENABLE_GLOBAL_HARDENING)
        include(cmake/Hardening.cmake)
        if (NOT SUPPORTS_UBSAN
                OR ASPH_ENABLE_SANITIZER_UNDEFINED
                OR ASPH_ENABLE_SANITIZER_ADDRESS
                OR ASPH_ENABLE_SANITIZER_THREAD
                OR ASPH_ENABLE_SANITIZER_LEAK)
            set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
        else ()
            set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
        endif ()
        sph_enable_hardening(sph_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
    endif ()

endmacro()
