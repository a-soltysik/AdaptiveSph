cmake_minimum_required(VERSION 3.16)

set(SRC_DIR ${imgui_SOURCE_DIR})
project(imgui CXX)

set(CMAKE_DEBUG_POSTFIX d)

if(APPLE)
    set(CMAKE_CXX_STANDARD 11)
    enable_language(OBJCXX)
endif()

add_library(${PROJECT_NAME})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(
    ${PROJECT_NAME}
    PRIVATE ${SRC_DIR}/imgui.cpp
            ${SRC_DIR}/imgui_demo.cpp
            ${SRC_DIR}/imgui_draw.cpp
            ${SRC_DIR}/imgui_tables.cpp
            ${SRC_DIR}/imgui_widgets.cpp
            ${SRC_DIR}/misc/cpp/imgui_stdlib.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

if(IMGUI_BUILD_ALLEGRO5_BINDING)
    find_package(unofficial-allegro5 CONFIG REQUIRED)
    target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE unofficial-allegro5::allegro
                unofficial-allegro5::allegro_ttf
                unofficial-allegro5::allegro_font
                unofficial-allegro5::allegro_main)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_allegro5.cpp)
endif()

if(IMGUI_BUILD_DX9_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_dx9.cpp)
endif()

if(IMGUI_BUILD_DX10_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_dx10.cpp)
endif()

if(IMGUI_BUILD_DX11_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_dx11.cpp)
endif()

if(IMGUI_BUILD_DX12_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_dx12.cpp)
endif()

if(IMGUI_BUILD_GLFW_BINDING)
    if(NOT EMSCRIPTEN)
        target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
    endif()
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_glfw.cpp)
endif()

if(IMGUI_BUILD_GLUT_BINDING)
    find_package(GLUT REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC GLUT::GLUT)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_glut.cpp)
endif()

if(IMGUI_BUILD_METAL_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_metal.mm)
    set_source_files_properties(${SRC_DIR}/backends/imgui_impl_metal.mm PROPERTIES COMPILE_FLAGS -fobjc-weak)
endif()

if(IMGUI_BUILD_OPENGL2_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_opengl2.cpp)
endif()

if(IMGUI_BUILD_OPENGL3_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_opengl3.cpp)
endif()

if(IMGUI_BUILD_OSX_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_osx.mm)
endif()

if(IMGUI_BUILD_SDL2_BINDING)
    find_package(SDL2 CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC SDL2::SDL2)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_sdl2.cpp)
endif()

if(IMGUI_BUILD_SDL2_RENDERER_BINDING)
    find_package(SDL2 CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC SDL2::SDL2)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_sdlrenderer2.cpp)
endif()

if(IMGUI_BUILD_VULKAN_BINDING)
    find_package(Vulkan REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_vulkan.cpp)
endif()

if(IMGUI_BUILD_WIN32_BINDING)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/backends/imgui_impl_win32.cpp)
endif()

if(IMGUI_FREETYPE)
    target_link_libraries(${PROJECT_NAME} PUBLIC freetype)
    target_sources(${PROJECT_NAME} PRIVATE ${SRC_DIR}/misc/freetype/imgui_freetype.cpp)
    target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_ENABLE_FREETYPE)
endif()

if(IMGUI_FREETYPE_LUNASVG)
    find_package(unofficial-lunasvg CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC unofficial::lunasvg::lunasvg)
    target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_ENABLE_FREETYPE_LUNASVG)
endif()

if(IMGUI_USE_WCHAR32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_USE_WCHAR32)
endif()

list(REMOVE_DUPLICATES BINDINGS_SOURCES)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_target
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

foreach(BINDING_TARGET ${BINDING_TARGETS})
    install(
        TARGETS ${BINDING_TARGET}
        EXPORT ${PROJECT_NAME}_target
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
endforeach()

if(NOT IMGUI_SKIP_HEADERS)
    install(
        FILES ${SRC_DIR}/imgui.h
              ${SRC_DIR}/imconfig.h
              ${SRC_DIR}/imgui_internal.h
              ${SRC_DIR}/imstb_textedit.h
              ${SRC_DIR}/imstb_rectpack.h
              ${SRC_DIR}/imstb_truetype.h
              ${SRC_DIR}/misc/cpp/imgui_stdlib.h
        DESTINATION include)

    if(IMGUI_BUILD_ALLEGRO5_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_allegro5.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_DX9_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_dx9.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_DX10_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_dx10.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_DX11_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_dx11.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_DX12_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_dx12.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_GLFW_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_glfw.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_GLUT_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_glut.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_METAL_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_metal.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_OPENGL2_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_opengl2.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_OPENGL3_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_opengl3.h ${SRC_DIR}/backends/imgui_impl_opengl3_loader.h
                DESTINATION include)
    endif()

    if(IMGUI_BUILD_OSX_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_osx.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_SDL2_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_sdl2.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_SDL2_RENDERER_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_sdlrenderer2.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_VULKAN_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_vulkan.h DESTINATION include)
    endif()

    if(IMGUI_BUILD_WIN32_BINDING)
        install(FILES ${SRC_DIR}/backends/imgui_impl_win32.h DESTINATION include)
    endif()

    if(IMGUI_FREETYPE)
        install(FILES ${SRC_DIR}/misc/freetype/imgui_freetype.h DESTINATION include)
    endif()
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/imgui-config.cmake DESTINATION share/imgui)

install(
    EXPORT ${PROJECT_NAME}_target
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}-targets.cmake
    DESTINATION share/${PROJECT_NAME})
