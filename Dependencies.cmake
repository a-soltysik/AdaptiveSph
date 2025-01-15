include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(sph_setup_dependencies)

    if(NOT TARGET fmtlib::fmtlib)
        cpmaddpackage("gh:fmtlib/fmt#11.1.1")
    endif()
    if(NOT TARGET PD::Engine)
        cpmaddpackage("gh:a-soltysik/Panda#1.0.2")
    endif()

endfunction()
