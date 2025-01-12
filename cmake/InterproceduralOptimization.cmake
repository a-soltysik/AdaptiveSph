macro(sph_enable_ipo)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif()
endmacro()

macro(sph_suppress_ipo)
    if(sph_ENABLE_IPO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
endmacro()

macro(sph_resume_ipo)
    if(sph_ENABLE_IPO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT result OUTPUT output)
        if(result)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
    endif()
endmacro()