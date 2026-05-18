from __future__ import annotations

import jax._src.core as jax_core
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import numpy

_cache: dict = {}

def _make_ir_attr( v ):
    if isinstance( v, ( int, numpy.integer ) ):
        return ir.IntegerAttr.get( ir.IntegerType.get_signless( 64 ), int( v ) )
    if isinstance( v, ( float, numpy.floating ) ):
        return ir.FloatAttr.get( ir.F64Type.get(), float( v ) )
    raise NotImplementedError( f"Unsupported FFI attribute type: { type( v ) }" )

def get_or_create( module_name: str, output_specs, attributes: dict = {} ) -> jax_core.Primitive:
    """
    Returns (creating if needed) a JAX Primitive whose MLIR lowering emits
    a stablehlo.custom_call targeting the XLA FFI handler `module_name`.
    output_specs : list[ jax.ShapeDtypeStruct ]
    The cache key includes output shapes and attribute values so different
    capacities or attribute combos get distinct primitives.
    """
    cache_key = ( module_name, tuple( ( tuple( s.shape ), s.dtype ) for s in output_specs ), tuple( sorted( attributes.items() ) ) )
    if cache_key in _cache:
        return _cache[ cache_key ]

    prim = jax_core.Primitive( module_name )
    prim.multiple_results = True

    def abstract_eval( *avals, **_ ):
        return [ jax_core.ShapedArray( s.shape, s.dtype ) for s in output_specs ]

    prim.def_abstract_eval( abstract_eval )

    def lower( ctx, *mlir_args, **_ ):
        out_types = [ mlir.aval_to_ir_type( a ) for a in ctx.avals_out ]
        backend_config_dict = { k: _make_ir_attr( v ) for k, v in attributes.items() }
        op = hlo.CustomCallOp(
            out_types,
            list( mlir_args ),
            call_target_name = ir.StringAttr.get( module_name ),
            api_version      = ir.IntegerAttr.get( ir.IntegerType.get_signless( 32 ), 4 ),
            backend_config   = ir.DictAttr.get( backend_config_dict ),
        )
        return op.results

    mlir.register_lowering( prim, lower )
    _cache[ cache_key ] = prim
    return prim
