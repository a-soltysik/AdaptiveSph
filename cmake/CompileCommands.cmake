function(sph_enable_compile_commands)
    set(CMAKE_EXPORT_COMPILE_COMMANDS
        ON
        CACHE INTERNAL "")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/compile_commands.json
                            ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endfunction()
