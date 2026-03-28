import sys, ctypes, sysconfig
ext = sysconfig.get_config_var('EXT_SUFFIX')[1:]
print( 'generated/bsp_FP64_2_cpu.' + ext )
lib = ctypes.CDLL( 'generated/bsp_FP64_2_cpu.' + ext )
init = lib.PyInit_bsp_FP64_2_cpu
init.restype = ctypes.py_object
try:
    m = init()
    print('PyInit retourne:', m, type(m))
except Exception as e:
    import traceback; traceback.print_exc()
