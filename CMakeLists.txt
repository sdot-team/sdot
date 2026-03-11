cmake_minimum_required(VERSION 3.15)
project(sdot_w2 CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---
if(APPLE)
    # Common Homebrew paths for libomp
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
endif()

# Prioritize virtual environment if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv")
    set(Python_FIND_VIRTUALENV FIRST)
    set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

find_package(OpenMP)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Determine the extension for Python modules
# FindPython provides Python_SOABI since 3.17
if(NOT Python_SOABI)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" OUTPUT_VARIABLE Python_MODULE_EXTENSION OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    set(Python_MODULE_EXTENSION ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

# Use the found Python to help find Torch
execute_process(COMMAND ${Python_EXECUTABLE} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths()))" OUTPUT_VARIABLE TORCH_INCLUDE_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.library_paths()[0])" OUTPUT_VARIABLE TORCH_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

# --- Core Library ---
include_directories(include)
add_library(sdot_cpu SHARED src/cpu/sdot_w2_cpu.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sdot_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    add_library(sdot_metal SHARED src/metal/sdot_w2_metal.mm)
    set_source_files_properties(src/metal/sdot_w2_metal.mm PROPERTIES LANGUAGE OBJCXX)
    target_link_libraries(sdot_metal PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
endif()

# --- PyTorch Extension ---
add_library(sdot_pytorch_cpp MODULE python/pytorch/sdot_pytorch_cpp.cpp)
target_compile_definitions(sdot_pytorch_cpp PRIVATE TORCH_EXTENSION_NAME=sdot_pytorch_cpp)
target_include_directories(sdot_pytorch_cpp PRIVATE include ${TORCH_INCLUDE_PATHS} ${Python_INCLUDE_DIRS})
target_link_directories(sdot_pytorch_cpp PRIVATE ${TORCH_LIB_PATH})
target_link_libraries(sdot_pytorch_cpp PRIVATE sdot_cpu torch torch_python torch_cpu c10)
set_target_properties(sdot_pytorch_cpp PROPERTIES 
    PREFIX "" 
    OUTPUT_NAME "sdot_pytorch_cpp"
    SUFFIX "${Python_MODULE_EXTENSION}"
)
if(APPLE)
    set_target_properties(sdot_pytorch_cpp PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

# --- JAX / Nanobind Extension ---
execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind REQUIRED)

nanobind_add_module(sdot_jax_cpp python/jax/sdot_jax_cpp.cpp)
target_link_libraries(sdot_jax_cpp PRIVATE sdot_cpu)
set_target_properties(sdot_jax_cpp PROPERTIES 
    PREFIX "" 
    SUFFIX "${Python_MODULE_EXTENSION}"
)

# --- C++ Tests ---
add_executable(test_w2_cpu tests/test_w2_cpu.cpp)
target_link_libraries(test_w2_cpu sdot_cpu)
