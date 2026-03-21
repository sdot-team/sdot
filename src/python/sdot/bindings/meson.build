py.extension_module( 'sdot_bindings_cpu',
  [ 'sdot_bindings_cpu.cpp' ],
  # include_directories: include_directories('../../../'), # points to 'src'
  dependencies: [ py_dep, nanobind_dep ],
  cpp_args: common_cpp_args,
  install: false
)

py.extension_module( 'sdot_bsp_bindings',
  [ 'sdot_bsp_bindings.cpp' ],
  # include_directories: include_directories('../../../'), # points to 'src'
  dependencies: [ py_dep, nanobind_dep, zpp_bits_dep ],
  cpp_args: common_cpp_args,
  install: false
)
