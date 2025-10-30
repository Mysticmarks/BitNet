cmake_minimum_required(VERSION 3.18)
project(bitnet_minimal_integration LANGUAGES CXX)

option(BITNET_INTEGRATION_ENABLE_GPU "Build GPU validation executable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bitnet_cpu_integration cpu_integration_test.cpp)

if (BITNET_INTEGRATION_ENABLE_GPU)
    enable_language(CUDA OPTIONAL)
    if (CMAKE_CUDA_COMPILER)
        add_executable(bitnet_gpu_integration gpu_integration_test.cu)
        set_target_properties(bitnet_gpu_integration PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 14
            CUDA_STANDARD_REQUIRED ON
        )
    endif()
endif()
