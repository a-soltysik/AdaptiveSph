function(sph_create_compile_commands_symlink)
    set(source_compile_commands "${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json")
    set(build_compile_commands "${CMAKE_BINARY_DIR}/compile_commands.json")

    if (IS_SYMLINK "${source_compile_commands}")
        file(READ_SYMLINK "${source_compile_commands}" current_target)
        if (NOT "${current_target}" STREQUAL "${build_compile_commands}")
            file(REMOVE "${source_compile_commands}")
        endif ()
    elseif (EXISTS "${source_compile_commands}")
        message(WARNING "Cannot create compile_commands.json symlink: a source-tree file already exists")
        return()
    endif ()

    if (NOT EXISTS "${source_compile_commands}" AND NOT IS_SYMLINK "${source_compile_commands}")
        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E create_symlink "${build_compile_commands}" "${source_compile_commands}"
                RESULT_VARIABLE symlink_result
                ERROR_VARIABLE symlink_error)
        if (NOT "${symlink_result}" STREQUAL "0")
            message(WARNING "Could not create compile_commands.json symlink: ${symlink_error}")
        endif ()
    endif ()
endfunction()
