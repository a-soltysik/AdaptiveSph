function(target_include_system_directories target)
    set(multiValueArgs INTERFACE PUBLIC PRIVATE)
    cmake_parse_arguments(
            ARG
            ""
            ""
            "${multiValueArgs}"
            ${ARGN})

    foreach (scope IN ITEMS INTERFACE PUBLIC PRIVATE)
        foreach (lib_include_dirs IN LISTS ARG_${scope})
            if (NOT MSVC)
                # system includes do not work in MSVC
                # awaiting https://gitlab.kitware.com/cmake/cmake/-/issues/18272#
                # awaiting https://gitlab.kitware.com/cmake/cmake/-/issues/17904
                set(_SYSTEM SYSTEM)
            endif ()
            if (${scope} STREQUAL "INTERFACE" OR ${scope} STREQUAL "PUBLIC")
                target_include_directories(
                        ${target}
                        ${_SYSTEM}
                        ${scope}
                        "$<BUILD_INTERFACE:${lib_include_dirs}>"
                        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
            else ()
                target_include_directories(
                        ${target}
                        ${_SYSTEM}
                        ${scope}
                        ${lib_include_dirs})
            endif ()
        endforeach ()
    endforeach ()

endfunction()

function(
        target_include_system_library
        target
        scope
        lib)
    if (TARGET ${lib})
        get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
        if (lib_include_dirs)
            target_include_system_directories(${target} ${scope} ${lib_include_dirs})
        else ()
            message(TRACE "${lib} library does not have the INTERFACE_INCLUDE_DIRECTORIES property.")
        endif ()
    endif ()
endfunction()

function(
        target_link_system_library
        target
        scope
        lib)
    target_include_system_library(${target} ${scope} ${lib})
    target_link_libraries(${target} ${scope} ${lib})
endfunction()

function(target_link_system_libraries target)
    set(multiValueArgs INTERFACE PUBLIC PRIVATE)
    cmake_parse_arguments(
            ARG
            ""
            ""
            "${multiValueArgs}"
            ${ARGN})

    foreach (scope IN ITEMS INTERFACE PUBLIC PRIVATE)
        foreach (lib IN LISTS ARG_${scope})
            target_link_system_library(${target} ${scope} ${lib})
        endforeach ()
    endforeach ()
endfunction()
