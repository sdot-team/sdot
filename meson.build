project('sdot', [ 'cpp', 'objcpp' ], version: '0.1.0', default_options: ['cpp_std=c++23'] )

# Python module import and find installation
py_mod = import( 'python' )
py = py_mod.find_installation( pure: false )

# Get Python dependency
py_dep = py.dependency()

# --- Nanobind Setup (for both PyTorch and JAX) ---
# Get nanobind include and source directories
nanobind_include_cmd = run_command( py, [ '-c', 'import nanobind; print(nanobind.include_dir())' ], check: true )
nanobind_include = nanobind_include_cmd.stdout().strip()

nanobind_src_cmd = run_command( py, [ '-c', 'import nanobind; import os; print(os.path.join(os.path.dirname(nanobind.__file__), "src", "nb_combined.cpp"))'], check: true )
nanobind_combined_src = nanobind_src_cmd.stdout().strip()

# Get robin_map include directory (dependency of nanobind)
robin_map_include_cmd = run_command( py, [ '-c', 'import nanobind; import os; print(os.path.join(os.path.dirname(nanobind.__file__), "ext", "robin_map", "include"))'], check: true )
robin_map_include = robin_map_include_cmd.stdout().strip()

# --- Core CPU Library ---
# sdot_cpu = static_library( 'sdot_cpu',
#   [
#     'src/read_arg_name.cpp',
#     'src/cpu/sdot_l2_cpu.cpp',
#     'src/cpu/sdot_w2_cpu.cpp'
#   ],
#   include_directories: include_directories('src')
# )

# --- Metal Library (macOS only) ---
# if host_machine.system() == 'darwin'
#   metal_dep = dependency( 'appleframeworks', modules: [ 'Metal', 'Foundation' ] )

#   sdot_metal = static_library( 'sdot_metal',
#     'src/metal/sdot_w2_metal.mm',
#     dependencies: metal_dep,
#     # link_with: sdot_cpu,
#     include_directories: include_directories( 'src' )
#   )
# endif

# --- PyTorch Extension (nanobind) ---
# PyTorch extension now uses nanobind for better interoperability
# sdot_pytorch_cpp = py.extension_module('sdot_pytorch_cpp',
#   [ 'python/pytorch/sdot_pytorch_cpp.cpp', nanobind_combined_src ],
#   dependencies: py_dep,
#   # link_with: sdot_cpu,
#   include_directories: include_directories( 'src' ),
#   cpp_args: [ '-fvisibility=hidden', '-I' + nanobind_include, '-I' + robin_map_include ],
#   install: false
# )

# --- JAX Extension (nanobind) ---
# sdot_jax_cpp = py.extension_module( 'sdot_jax_cpp',
#   ['python/jax/sdot_jax_cpp.cpp', nanobind_combined_src],
#   dependencies: py_dep,
#   # link_with: sdot_cpu,
#   include_directories: include_directories('src'),
#   cpp_args: ['-fvisibility=hidden', '-I' + nanobind_include, '-I' + robin_map_include],
#   install: false
# )

# --- Tests with Catch2 ---
catch2_dep = dependency( 'catch2-with-main', version: '>=3.0.0', required: false )

if not catch2_dep.found()
  catch2_proj = subproject('catch2', default_options: ['cpp_std=c++20', 'tests=false'])
  catch2_dep = catch2_proj.get_variable('catch2_with_main_dep')
endif

# if host_machine.system() == 'darwin'
#   # Metal tests not yet implemented
#   # test_sources += files('tests/test_metal.cpp')
#   test_link = sdot_metal
# else
#   test_link = sdot_cpu
# endif

unit_tests = executable('unit_tests',
  [ 'tests/test_w2_cpu.cpp', ],
  dependencies: catch2_dep,
  include_directories: include_directories('src'),
  # Force absolute paths for clickable errors in Zed, and disable hardening
  cpp_args: [
    '-fdiagnostics-absolute-paths',
    '-U_LIBCPP_HARDENING_MODE',
    '-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE'
  ]
)

test( 'C++ Unit Tests', unit_tests )
