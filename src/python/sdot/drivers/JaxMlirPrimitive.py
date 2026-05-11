from __future__ import annotations

import jax._src.core as jax_core
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

_cache: dict[ str, jax_core.Primitive ] = {}

def get_or_create( module_name: str, output_specs ) -> jax_core.Primitive:
    """
    Returns (creating if needed) a JAX Primitive whose MLIR lowering emits
    a stablehlo.custom_call targeting the XLA FFI handler `module_name`.
    output_specs : list[ jax.ShapeDtypeStruct ]
    """
    if module_name in _cache:
        return _cache[ module_name ]

    prim = jax_core.Primitive( module_name )
    prim.multiple_results = True

    def abstract_eval( *avals, **_ ):
        return [ jax_core.ShapedArray( s.shape, s.dtype ) for s in output_specs ]

    prim.def_abstract_eval( abstract_eval )

    def lower( ctx, *mlir_args, **_ ):
        out_types = [ mlir.aval_to_ir_type( a ) for a in ctx.avals_out ]
        op = hlo.CustomCallOp(
            out_types,
            list( mlir_args ),
            call_target_name = ir.StringAttr.get( module_name ),
            api_version      = ir.IntegerAttr.get( ir.IntegerType.get_signless( 32 ), 4 ),
            backend_config   = ir.DictAttr.get( {} ),
        )
        return op.results

    mlir.register_lowering( prim, lower )
    _cache[ module_name ] = prim
    return prim
