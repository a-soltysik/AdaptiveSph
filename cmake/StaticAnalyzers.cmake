macro(sph_enable_cppcheck WARNINGS_AS_ERRORS)
    find_program(CPPCHECK cppcheck)
    if (CPPCHECK)

        if (CMAKE_GENERATOR MATCHES ".*Visual Studio.*")
            set(CPPCHECK_TEMPLATE "vs")
        else ()
            set(CPPCHECK_TEMPLATE "gcc")
        endif ()

        if ("${CPPCHECK_OPTIONS}" STREQUAL "")
            set(SUPPRESS_DIR "*:${CMAKE_CURRENT_BINARY_DIR}/_deps/*.h")
            message(STATUS "CPPCHECK_OPTIONS suppress: ${SUPPRESS_DIR}")
            set(CMAKE_CXX_CPPCHECK
                    ${CPPCHECK}
                    --template=${CPPCHECK_TEMPLATE}
                    --enable=warning,style,performance,portability,missingInclude
                    --inline-suppr
                    --suppress=cppcheckError
                    --suppress=missingIncludeSystem
                    --suppress=internalAstError
                    --suppress=unmatchedSuppression
                    --suppress=unknownMacro
                    --inconclusive
                    --check-level=exhaustive
                    --suppress=${SUPPRESS_DIR}
                    --std=c++${CMAKE_CXX_STANDARD})
        endif ()

        if (${WARNINGS_AS_ERRORS})
            list(APPEND CMAKE_CXX_CPPCHECK --error-exitcode=1)
        endif ()
    else ()
        message(${WARNING_MESSAGE} "cppcheck requested but executable not found")
    endif ()
endmacro()

macro(sph_enable_clang_tidy target WARNINGS_AS_ERRORS)

    find_program(CLANGTIDY clang-tidy)
    if (CLANGTIDY)
        if (NOT
                CMAKE_CXX_COMPILER_ID
                MATCHES
                ".*Clang")

            get_target_property(TARGET_PCH ${target} INTERFACE_PRECOMPILE_HEADERS)

            if ("${TARGET_PCH}" STREQUAL "TARGET_PCH-NOTFOUND")
                get_target_property(TARGET_PCH ${target} PRECOMPILE_HEADERS)
            endif ()

            if (NOT ("${TARGET_PCH}" STREQUAL "TARGET_PCH-NOTFOUND"))
                message(
                        SEND_ERROR
                        "clang-tidy cannot be enabled with non-clang compiler and PCH, clang-tidy fails to handle gcc's PCH file"
                )
            endif ()
        endif ()

        set(CLANG_TIDY_OPTIONS
                ${CLANGTIDY}
                -extra-arg=-Wno-unknown-warning-option
                -extra-arg=-Wno-ignored-optimization-argument
                -extra-arg=-Wno-unused-command-line-argument
                -p)
        if (NOT
                "${CMAKE_CXX_STANDARD}"
                STREQUAL
                "")
            if ("${CLANG_TIDY_OPTIONS_DRIVER_MODE}" STREQUAL "cl")
                set(CLANG_TIDY_OPTIONS ${CLANG_TIDY_OPTIONS} -extra-arg=/std:c++${CMAKE_CXX_STANDARD})
            else ()
                set(CLANG_TIDY_OPTIONS ${CLANG_TIDY_OPTIONS} -extra-arg=-std=c++${CMAKE_CXX_STANDARD})
            endif ()
        endif ()

        if (NOT ${WARNINGS_AS_ERRORS})
            list(APPEND CLANG_TIDY_OPTIONS -warnings-as-errors='')
        endif ()

        message("Also setting clang-tidy globally")
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_OPTIONS})
    else ()
        message(${WARNING_MESSAGE} "clang-tidy requested but executable not found")
    endif ()
endmacro()

macro(sph_enable_include_what_you_use)
    find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
    if (INCLUDE_WHAT_YOU_USE)
        set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE})
    else ()
        message(${WARNING_MESSAGE} "include-what-you-use requested but executable not found")
    endif ()
endmacro()
