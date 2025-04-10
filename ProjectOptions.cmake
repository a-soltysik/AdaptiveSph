include(cmake/SystemLink.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

macro(sph_supports_sanitizers)
    if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
        set(SUPPORTS_UBSAN ON)
    else()
        set(SUPPORTS_UBSAN OFF)
    endif()

    if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
        set(SUPPORTS_ASAN OFF)
    else()
        set(SUPPORTS_ASAN ON)
    endif()
endmacro()

macro(sph_setup_options)
    option(sph_ENABLE_HARDENING "Enable hardening" ON)
    cmake_dependent_option(
        sph_ENABLE_GLOBAL_HARDENING
        "Attempt to push hardening options to built dependencies"
        ON
        sph_ENABLE_HARDENING
        OFF)

    sph_supports_sanitizers()

    if(NOT PROJECT_IS_TOP_LEVEL OR sph_PACKAGING_MAINTAINER_MODE)
        option(sph_ENABLE_IPO "Enable IPO/LTO" OFF)
        option(sph_ENABLE_WARNINGS "Enable warnings" OFF)
        option(sph_ENABLE_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
        option(sph_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
        option(sph_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
        option(sph_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
        option(sph_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
        option(sph_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
        option(sph_ENABLE_IWYU "Enable include-what-you-use analysis" OFF)
        option(sph_ENABLE_PCH "Enable precompiled headers" OFF)
        option(sph_ENABLE_CACHE "Enable ccache" OFF)
        option(sph_ENABLE_COMPILE_COMMANDS "Enable support for compile_commnads.json" OFF)
        option(sph_ENABLE_FAST_MATH "Enable fast math compilation flags" OFF)
    else()
        option(sph_ENABLE_IPO "Enable IPO/LTO" OFF)
        option(sph_ENABLE_WARNINGS "Enable warnings" ON)
        option(sph_ENABLE_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
        option(sph_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
        option(sph_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
        option(sph_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
        option(sph_ENABLE_UNITY_BUILD "Enable unity builds" ON)
        option(sph_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
        option(sph_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
        option(sph_ENABLE_IWYU "Enable include-what-you-use analysis" OFF)
        option(sph_ENABLE_PCH "Enable precompiled headers" OFF)
        option(sph_ENABLE_CACHE "Enable ccache" OFF)
        option(sph_ENABLE_COMPILE_COMMANDS "Enable support for compile_commnads.json" ON)
        option(sph_ENABLE_FAST_MATH "Enable fast math compilation flags" OFF)
    endif()

    if(NOT PROJECT_IS_TOP_LEVEL)
        mark_as_advanced(
            sph_ENABLE_IPO
            sph_ENABLE_WARNINGS
            sph_ENABLE_WARNINGS_AS_ERRORS
            sph_ENABLE_USER_LINKER
            sph_ENABLE_SANITIZER_ADDRESS
            sph_ENABLE_SANITIZER_LEAK
            sph_ENABLE_SANITIZER_UNDEFINED
            sph_ENABLE_SANITIZER_THREAD
            sph_ENABLE_SANITIZER_MEMORY
            sph_ENABLE_UNITY_BUILD
            sph_ENABLE_CLANG_TIDY
            sph_ENABLE_CPPCHECK
            sph_ENABLE_IWYU
            sph_ENABLE_PCH
            sph_ENABLE_CACHE
            sph_ENABLE_COMPILE_COMMANDS
            sph_ENABLE_FAST_MATH)
    endif()

endmacro()

macro(sph_global_options)
    if(sph_ENABLE_IPO)
        include(cmake/InterproceduralOptimization.cmake)
        sph_enable_ipo()
    endif()

    sph_supports_sanitizers()

    if(sph_ENABLE_HARDENING AND sph_ENABLE_GLOBAL_HARDENING)
        include(cmake/Hardening.cmake)
        if(NOT SUPPORTS_UBSAN
           OR sph_ENABLE_SANITIZER_UNDEFINED
           OR sph_ENABLE_SANITIZER_ADDRESS
           OR sph_ENABLE_SANITIZER_THREAD
           OR sph_ENABLE_SANITIZER_LEAK)
            set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
        else()
            set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
        endif()
        sph_enable_hardening(sph_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
    endif()
endmacro()

macro(sph_local_options)
    if(PROJECT_IS_TOP_LEVEL)
        include(cmake/StandardProjectSettings.cmake)
    endif()

    if(sph_ENABLE_COMPILE_COMMANDS)
        include(cmake/CompileCommands.cmake)
        sph_enable_compile_commands()
    endif()

    add_library(sph_warnings INTERFACE)
    add_library(sph_options INTERFACE)

    if(sph_ENABLE_WARNINGS)
        include(cmake/CompilerWarnings.cmake)
        sph_set_project_warnings(
            sph_warnings
            ${sph_ENABLE_WARNINGS_AS_ERRORS}
            ""
            ""
            ""
            "")
    endif()

    if(sph_ENABLE_USER_LINKER)
        include(cmake/Linker.cmake)
        sph_configure_linker(sph_options)
    endif()

    include(cmake/Sanitizers.cmake)
    sph_enable_sanitizers(
        sph_options
        ${sph_ENABLE_SANITIZER_ADDRESS}
        ${sph_ENABLE_SANITIZER_LEAK}
        ${sph_ENABLE_SANITIZER_UNDEFINED}
        ${sph_ENABLE_SANITIZER_THREAD}
        ${sph_ENABLE_SANITIZER_MEMORY})

    set_target_properties(sph_options PROPERTIES UNITY_BUILD ${sph_ENABLE_UNITY_BUILD})

    if(sph_ENABLE_PCH)
        target_precompile_headers(
            sph_options
            INTERFACE
            <vector>
            <string>
            <utility>)
    endif()

    if(sph_ENABLE_CACHE)
        include(cmake/Cache.cmake)
        sph_enable_cache()
    endif()

    include(cmake/StaticAnalyzers.cmake)
    if(sph_ENABLE_CLANG_TIDY)
        sph_enable_clang_tidy(sph_options ${sph_ENABLE_WARNINGS_AS_ERRORS})
    endif()

    if(sph_ENABLE_CPPCHECK)
        sph_enable_cppcheck(${sph_ENABLE_WARNINGS_AS_ERRORS} "" # override cppcheck options
        )
    endif()

    if(sph_ENABLE_IWYU)
        sph_enable_include_what_you_use()
    endif()

    if(sph_ENABLE_FAST_MATH)
        include(cmake/Optimizations.cmake)
        sph_enable_fast_math(sph_options)
    endif()

    if(sph_ENABLE_HARDENING AND NOT sph_ENABLE_GLOBAL_HARDENING)
        include(cmake/Hardening.cmake)
        if(NOT SUPPORTS_UBSAN
           OR sph_ENABLE_SANITIZER_UNDEFINED
           OR sph_ENABLE_SANITIZER_ADDRESS
           OR sph_ENABLE_SANITIZER_THREAD
           OR sph_ENABLE_SANITIZER_LEAK)
            set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
        else()
            set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
        endif()
        sph_enable_hardening(sph_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
    endif()

endmacro()
