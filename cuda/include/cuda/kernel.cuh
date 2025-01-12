#ifdef _WIN32
#    ifdef EXPORTING_CUDA_LIB
#        define SPH_CUDA_API __declspec(dllexport)
#    else
#        define SPH_CUDA_API __declspec(dllimport)
#    endif
#else
#    define SPH_CUDA_API
#endif

SPH_CUDA_API void helloWorld();