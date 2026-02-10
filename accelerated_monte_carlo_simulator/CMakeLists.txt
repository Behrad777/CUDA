cmake_minimum_required(VERSION 3.18)

project(monte_carlo_cuda LANGUAGES CXX CUDA)

# --- Language standards ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Enable CUDA separable compilation (safe default) ---
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# --- Executable ---
add_executable(monte_carlo
    src/main.cpp
    src/cuda/kernels.cu
    src/cuda/rng_curand.cu
)

# --- Include paths ---
target_include_directories(monte_carlo PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# --- CUDA properties ---
set_target_properties(monte_carlo PROPERTIES
    CUDA_ARCHITECTURES native
)

# --- Link CUDA runtime (usually automatic, but explicit is fine) ---
target_link_libraries(monte_carlo
    cuda
)

# --- Optional: better warnings ---
target_compile_options(monte_carlo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
