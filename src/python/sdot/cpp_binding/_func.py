"""CppFunc descriptor and per-function module cache."""

from ._build import get_module_for, CppFunc

_func_cache: dict = {}


def get_func_for( func_name, args, includes ) -> callable:
    func = CppFunc( func_name, args, includes )
    id = func.key()

    if id in _func_cache:
        return _func_cache[ id ]

    module = get_module_for( [ func ] )

    res = getattr( module, func_name )
    _func_cache[ id ] = res
    return res

def get_forward_and_backward_for( func_name, args, includes, input_tensors, output_tensors ) -> tuple:
    func_bwd = CppFunc( func_name + "_backward", list( args ) + input_tensors + output_tensors, includes )
    func_fwd = CppFunc( func_name, args, includes )

    id = ( func_fwd.key(), "with_backward" )
    if id in _func_cache:
        return _func_cache[ id ]

    module = get_module_for( [ func_fwd, func_bwd ] )

    res = ( getattr( module, func_name ), getattr( module, func_name + "_backward" ) )
    _func_cache[ id ] = res
    return res
