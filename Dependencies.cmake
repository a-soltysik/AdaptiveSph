function(sph_setup_dependencies)

    CPMAddPackage(
            URI "gh:fmtlib/fmt#11.1.1"
            OPTIONS "FMT_UNICODE OFF")
    CPMAddPackage(
            URI "gh:glfw/glfw#3.4"
            OPTIONS "GLFW_BUILD_DOCS OFF")
    CPMAddPackage("gh:g-truc/glm#1.0.1")
    CPMAddPackage("gh:nlohmann/json@3.12.0")
    CPMAddPackage(
            URI "gh:jlblancoc/nanoflann@1.8.0"
            OPTIONS
            "MASTER_PROJECT_HAS_TARGET_UNINSTALL ON")
    CPMAddPackage(
            NAME
            ctre
            VERSION
            3.9.0
            GITHUB_REPOSITORY
            hanickadot/compile-time-regular-expressions)

    set(IMGUI_BUILD_GLFW_BINDING ON)
    set(IMGUI_BUILD_VULKAN_BINDING ON)

    CPMAddPackage("gh:ocornut/imgui@1.91.6")
    add_subdirectory(ext/imgui-cmake)

endfunction()
