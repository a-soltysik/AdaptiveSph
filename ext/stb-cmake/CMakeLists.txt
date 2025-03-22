cmake_minimum_required(VERSION 3.16)
project(stb C)

set(CMAKE_DEBUG_POSTFIX d)

add_library(${PROJECT_NAME}_image INTERFACE)
add_library(${PROJECT_NAME}::image ALIAS ${PROJECT_NAME}_image)
target_include_directories(${PROJECT_NAME}_image INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                           $<INSTALL_INTERFACE:include>)

target_compile_definitions(stb_image INTERFACE STBI_NO_SIMD)

install(
    TARGETS ${PROJECT_NAME}_image
    EXPORT ${PROJECT_NAME}_image_target
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/stb_image.h DESTINATION include)
