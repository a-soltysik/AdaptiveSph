set(SHADER_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SHADER_TARGET_NAME AdaptiveSphShader)

file(
    GLOB_RECURSE
    ${SHADER_TARGET_NAME}_SRC_LIST
    ${SHADER_SRC_DIR}/*.vert
    ${SHADER_SRC_DIR}/*.tesc
    ${SHADER_SRC_DIR}/*.tese
    ${SHADER_SRC_DIR}/*.geom
    ${SHADER_SRC_DIR}/*.frag
    ${SHADER_SRC_DIR}/*.comp)

foreach(GLSL_FILE ${${SHADER_TARGET_NAME}_SRC_LIST})
    get_filename_component(FILE_NAME ${GLSL_FILE} NAME)
    set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/shader)
    set(SPIRV "${OUTPUT_DIR}/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${GLSL_FILE} -o ${SPIRV}
        DEPENDS ${GLSL_FILE})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL_FILE)

add_custom_target(${SHADER_TARGET_NAME} DEPENDS ${SPIRV_BINARY_FILES})

add_dependencies(AdaptiveSphEngine ${SHADER_TARGET_NAME})
