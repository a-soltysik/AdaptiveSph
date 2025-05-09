include(cmake/CPM.cmake)

function(sph_setup_dependencies)

    if (NOT TARGET fmtlib::fmtlib)
        cpmaddpackage("gh:fmtlib/fmt#11.1.1")
    endif ()
    if (NOT TARGET glfw)
        cpmaddpackage("gh:glfw/glfw#3.4")
    endif ()
    if (NOT TARGET glm)
        cpmaddpackage("gh:g-truc/glm#1.0.1")
    endif ()
    if (NOT TARGET nlohmann_json)
        cpmaddpackage("gh:nlohmann/json@3.12.0")
    endif ()
    if (NOT TARGET ctre)
        cpmaddpackage(
                NAME
                ctre
                VERSION
                3.9.0
                GITHUB_REPOSITORY
                hanickadot/compile-time-regular-expressions)
    endif ()
    if (NOT TARGET imgui)
        set(IMGUI_BUILD_GLFW_BINDING ON)
        set(IMGUI_BUILD_VULKAN_BINDING ON)

        cpmaddpackage("gh:ocornut/imgui@1.91.6")
        add_subdirectory(ext/imgui-cmake)
    endif ()

endfunction()
