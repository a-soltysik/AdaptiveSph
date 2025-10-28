include("${CMAKE_CURRENT_LIST_DIR}/Utilities.cmake")

macro(detect_architecture)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CMAKE_SYSTEM_PROCESSOR_LOWER)
    if (CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL x86 OR CMAKE_SYSTEM_PROCESSOR_LOWER MATCHES "^i[3456]86$")
        set(VCVARSALL_ARCH x86)
    elseif (
            CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL x64
            OR CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL x86_64
            OR CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL amd64)
        set(VCVARSALL_ARCH x64)
    elseif (CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL arm)
        set(VCVARSALL_ARCH arm)
    elseif (CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL arm64 OR CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL aarch64)
        set(VCVARSALL_ARCH arm64)
    else ()
        if (CMAKE_HOST_SYSTEM_PROCESSOR)
            set(VCVARSALL_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
        else ()
            set(VCVARSALL_ARCH x64)
            message(STATUS "Unknown architecture CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR_LOWER} - using x64")
        endif ()
    endif ()
endmacro()

function(run_vcvarsall)
    if (MSVC AND "$ENV{VSCMD_VER}" STREQUAL "")

        get_filename_component(MSVC_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
        find_file(
                VCVARSALL_FILE
                NAMES vcvarsall.bat
                PATHS "${MSVC_DIR}"
                "${MSVC_DIR}/.."
                "${MSVC_DIR}/../.."
                "${MSVC_DIR}/../../../../../../../.."
                "${MSVC_DIR}/../../../../../../.."
                PATH_SUFFIXES "VC/Auxiliary/Build" "Common7/Tools" "Tools")

        if (EXISTS ${VCVARSALL_FILE})
            detect_architecture()
            message(STATUS "Running `${VCVARSALL_FILE} ${VCVARSALL_ARCH}` to set up the MSVC environment")
            execute_process(
                    COMMAND
                    "cmd" "/c" ${VCVARSALL_FILE} ${VCVARSALL_ARCH} #
                    "&&" "call" "echo" "VCVARSALL_ENV_START" #
                    "&" "set" #
                    OUTPUT_VARIABLE VCVARSALL_OUTPUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

            find_substring_by_prefix(VCVARSALL_ENV "VCVARSALL_ENV_START" "${VCVARSALL_OUTPUT}")
            set_env_from_string("${VCVARSALL_ENV}")

        else ()
            message(
                    WARNING
                    "Could not find `vcvarsall.bat` for automatic MSVC environment preparation. Please manually open the MSVC command prompt and rebuild the project.
      ")
        endif ()
    endif ()
endfunction()
