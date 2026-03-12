cmake_minimum_required(VERSION 3.15)
project(sdot_w2 CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force color output for compilers (Clang/GCC)
if(NOT MSVC)
    add_compile_options(-fdiagnostics-color=always)
endif()

# Catch2 configuration
set(CATCH_CONFIG_COLOUR_ANSI ON CACHE BOOL "Force ANSI color for Catch2")

# --- Dependencies ---
if(APPLE)
    # Prioritize torch's libomp if available to avoid loading multiple OpenMP runtimes
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" OUTPUT_VARIABLE TORCH_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${TORCH_LIB_DIR}/libomp.dylib")
        set(OpenMP_libomp_LIBRARY "${TORCH_LIB_DIR}/libomp.dylib" CACHE FILEPATH "Path to libomp.dylib")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${TORCH_LIB_DIR}/libomp.dylib")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB_DIR}")
        message(STATUS "Using PyTorch's OpenMP: ${TORCH_LIB_DIR}/libomp.dylib")
    endif()

    # Common Homebrew paths for libomp
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
endif()

include(FetchContent)

# Prioritize virtual environment if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv")
    set(Python_FIND_VIRTUALENV FIRST)
    set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

find_package(OpenMP)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Catch2 v3
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

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
add_library(sdot_cpu SHARED
    src/cpu/sdot_w2_cpu.cpp
    src/cpu/sdot_l2_cpu.cpp
    src/read_arg_name.cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sdot_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    add_library(sdot_metal SHARED
        src/metal/sdot_w2_metal.mm
        src/metal/sdot_l2_metal.mm
    )
    set_source_files_properties(src/metal/sdot_w2_metal.mm PROPERTIES LANGUAGE OBJCXX)
    set_source_files_properties(src/metal/sdot_l2_metal.mm PROPERTIES LANGUAGE OBJCXX)
    target_link_libraries(sdot_metal PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY} sdot_cpu)
endif()

# --- PyTorch Extension ---
add_library(sdot_pytorch_cpp MODULE python/pytorch/sdot_pytorch_cpp.cpp)
target_compile_definitions(sdot_pytorch_cpp PRIVATE TORCH_EXTENSION_NAME=sdot_pytorch_cpp)
target_include_directories(sdot_pytorch_cpp PRIVATE src/cpu ${TORCH_INCLUDE_PATHS} ${Python_INCLUDE_DIRS})
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
add_executable(unit_tests
    tests/test_w2_cpu.cpp
    tests/test_cpu.cpp
)
if(APPLE)
    target_sources(unit_tests PRIVATE tests/test_metal.cpp)
    target_link_libraries(unit_tests PRIVATE sdot_metal)
endif()
target_link_libraries(unit_tests PRIVATE sdot_cpu Catch2::Catch2WithMain)
add_test(NAME unit_tests COMMAND unit_tests)
