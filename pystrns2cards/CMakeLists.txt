cmake_minimum_required(VERSION 3.21)
project(pystrns2cards LANGUAGES CXX CUDA)

# === General Settings ===

# Position-independent code required for shared libs / Python modules
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release build unless specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Use C++14 for CUDA compatibility
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place final Python module (_core.pyd/.so) in this directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_LIST_DIR})

# === Python Version Configuration ===

# Option to allow overriding Python version; default is 3.10
set(PYTHON_VERSION_HINT "3.10" CACHE STRING "Python version to use")
find_package(Python ${PYTHON_VERSION_HINT} EXACT COMPONENTS Interpreter Development REQUIRED)

# === Nanobind ===
add_subdirectory(ext/nanobind)

# === CUDA + C++ Sources ===
set(SRC
    src/pystrns2cards.cpp
    src/antialias.cu
    src/bitset_utils.cu
    src/compact_utils.cu
    src/cuda_common.cuh
    src/curve_utils.cu
    src/dda.cu
    src/hair_loader.h
    src/hair_loader.cpp
    src/mesh_utils.h
    src/mesh_utils.cpp
    src/rasterize.cu
)

# === Nanobind Python Module (_core) ===

# Compiles _core module with both core.cpp and CUDA/C++ sources
nanobind_add_module(_core NOMINSIZE ${SRC})

# Include headers from src/ in both _core and nanobind bindings
target_include_directories(_core PUBLIC src)

# Enable separable compilation for CUDA
set_property(TARGET _core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
