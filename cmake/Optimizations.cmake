macro(sph_enable_fast_math target)

    message(STATUS "** Fast Math (Target ${target}) **")

    if(MSVC)
        target_compile_options(${target} INTERFACE $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/fp:fast>)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang|GNU")
        target_compile_options(${target} INTERFACE $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:-ffast-math>)
    endif()

    target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endmacro()
