project('sdot', [ 'cpp', 'objcpp' ], version: '0.1.0', default_options: ['cpp_std=c++20'] )

# Python module import and find installation
py_mod = import( 'python' )
py = py_mod.find_installation( pure: false )
py_dep = py.dependency()

# --- Nanobind Setup ---
nanobind_info_cmd = run_command( py, [ '-c',  '''
import nanobind, os; d = os.path.dirname(nanobind.__file__)
print(nanobind.include_dir())
print(os.path.join(d, "src", "nb_combined.cpp"))
print(os.path.join(d, "ext", "robin_map", "include"))
''' ], check: true )

nanobind_info = nanobind_info_cmd.stdout().strip().split('\n')
nanobind_combined_src = nanobind_info[1]
robin_map_include = nanobind_info[2]
nanobind_include = nanobind_info[0]

nanobind_dep = declare_dependency(
  compile_args: [ '-I' + nanobind_include, '-I' + robin_map_include, '-fvisibility=hidden' ],
  sources: [ nanobind_combined_src ]
)

# --- Common Project Settings ---
common_cpp_args = [ '-fdiagnostics-absolute-paths' ]

# --- Python extension modules ---
subdir('src/python/sdot/bindings')

# --- Tests with Catch2 ---
catch2_dep = dependency( 'catch2-with-main', version: '>=3.0.0', fallback: ['catch2', 'catch2_with_main_dep'] )

unit_tests = executable('unit_tests',
  [ 'tests/test_w2_cpu.cpp', ],
  dependencies: catch2_dep,
  include_directories: include_directories('src'),
  # Force absolute paths for clickable errors in Zed, and disable hardening
  # cpp_args: common_cpp_args + [
  #   '-U_LIBCPP_HARDENING_MODE',
  #   '-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE'
  # ]
)

test( 'C++ Unit Tests', unit_tests )
