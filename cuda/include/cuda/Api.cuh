#pragma once

#ifdef _WIN32
#    ifdef ASPH_CUDA_EXPORTS
#        define SPH_CUDA_API __declspec(dllexport)
#    else
#        define SPH_CUDA_API __declspec(dllimport)
#    endif
#else
#    define SPH_CUDA_API
#endif
