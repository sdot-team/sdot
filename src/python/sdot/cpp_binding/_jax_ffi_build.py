"""XLA FFI C++ code generator for sdot.

Generates, compiles, and caches nanobind modules that expose XLA FFI handler
capsules consumed by ``jax.ffi.ffi_call``.

Public entry point
------------------
``get_ffi_module(func_name, includes, arg_specs)``

arg_specs is a list of ``FfiArgSpec`` objects describing each argument passed
to the C++ function.
"""

from ._build import _compile, _try_import, force_build, _module_name
from sdot.drivers.compilation.encode_base_62  import encode_base62

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import importlib


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------

@dataclass
class BufferSpec:
    """One XLA buffer (= one TensorField or one plain array)."""
    ndim    : int
    dtype   : str   # 'float' | 'int'


@dataclass
class FfiArgSpec:
    """Describes one logical argument to the generated C++ function.

    kind:
      'mutable'      — existing object; buffers read AND written (Cell, etc.)
      'return_obj'   — new object produced by C++; buffers only written
      'return_tensor'— raw array produced by C++; single buffer written
      'array'        — plain read-only array input
      'scalar_int'   — int scalar, passed as int64_t attribute
      'scalar_float' — float scalar, passed as double attribute

    For mutable/return_obj: ``buffers`` lists all TensorField buffers in order.
    For array: ``buffers`` has exactly one entry.
    For return_tensor: ``buffers`` has exactly one entry.
    For scalars: ``buffers`` is empty.

    cpp_ctor: C++ constructor template for reconstructing the object.
      It receives positional %s placeholders for each buffer expression.
      e.g. "Cell<TF,%d,Cpu>(%s, %s, %s, %s)"
      If empty, the single buffer expression is used directly (array case).
    """
    kind      : str
    buffers   : list[ BufferSpec ] = field( default_factory=list )
    cpp_ctor  : str = ""          # only used for mutable/return_obj


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_module_cache: dict = {}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_ffi_module( func_name: str, includes: list[ str ], arg_specs: list[ FfiArgSpec ] ):
    """Return (compiling if necessary) a nanobind module that exposes
    ``fwd_capsule()`` (and optionally ``bwd_capsule()``) for *func_name*.
    """
    key = _make_key( func_name, includes, arg_specs )

    if key in _module_cache:
        return _module_cache[ key ]

    if not force_build():
        if res := _try_import( key ):
            _module_cache[ key ] = res
            return res

    src = _generate_source( func_name, includes, arg_specs, key )
    _write_and_compile( key, src )

    importlib.invalidate_caches()
    res = importlib.import_module( _module_name( key ) )
    _module_cache[ key ] = res
    return res


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------

def _make_key( func_name, includes, arg_specs ):
    parts = [ func_name ]
    for inc in includes:
        parts.append( inc.replace( "/", "_" ).replace( ".", "_" ) )
    for s in arg_specs:
        parts.append( s.kind[ :3 ] )
        for b in s.buffers:
            parts.append( f"{ b.ndim }{ b.dtype[ 0 ] }" )
    raw = "_".join( parts )
    if len( raw ) > 40:
        raw = raw[ :29 ] + encode_base62( raw[ 29: ] )
    return "ffi_" + raw


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _dtype_cpp( dtype: str ) -> str:
    return "SI" if dtype == "int" else "TF"


def _generate_source( func_name: str, includes: list[ str ], arg_specs: list[ FfiArgSpec ], module_name: str ) -> str:
    lines = []

    # --- includes ---
    std_includes = [
        "sdot/ffi_wrappers.h",
        "sdot/nanobind_wrappers.h",
        "nanobind/stl/optional.h",
        "nanobind/stl/vector.h",
        "nanobind/nanobind.h",
    ]
    for inc in std_includes + includes:
        lines.append( f"#include <{ inc }>" )
    lines.append( "" )
    lines.append( "namespace nb = nanobind;" )
    lines.append( "using namespace sdot;" )
    lines.append( "namespace ffi = xla::ffi;" )
    lines.append( "" )
    lines.append( "using TF = double;" )
    lines.append( "using SI = std::int64_t;" )
    lines.append( "" )

    # --- forward handler ---
    lines += _generate_handler( func_name, arg_specs )
    lines.append( "" )

    # --- backward handler ---
    lines += _generate_bwd_handler( func_name, arg_specs )
    lines.append( "" )

    # --- nanobind module ---
    lines.append( f"NB_MODULE( { module_name }, m ) {{" )
    lines.append( f"    m.def( \"fwd_capsule\", []() {{" )
    lines.append( f"        return nb::capsule( reinterpret_cast<void*>( &fwd_handler_{ func_name } )," )
    lines.append( f"                            \"xla._CUSTOM_CALL_TARGET\" );" )
    lines.append( f"    }} );" )
    lines.append( f"    m.def( \"bwd_capsule\", []() {{" )
    lines.append( f"        return nb::capsule( reinterpret_cast<void*>( &bwd_handler_{ func_name } )," )
    lines.append( f"                            \"xla._CUSTOM_CALL_TARGET\" );" )
    lines.append( f"    }} );" )
    lines.append( "}" )

    return "\n".join( lines ) + "\n"


def _generate_handler( func_name: str, arg_specs: list[ FfiArgSpec ] ) -> list[ str ]:
    """Generate the XLA FFI handler function + XLA_FFI_DEFINE_HANDLER_SYMBOL call."""

    mutable_specs  = [ s for s in arg_specs if s.kind == 'mutable' ]
    ret_obj_specs  = [ s for s in arg_specs if s.kind == 'return_obj' ]
    ret_tn_specs   = [ s for s in arg_specs if s.kind == 'return_tensor' ]
    array_specs    = [ s for s in arg_specs if s.kind == 'array' ]
    sint_specs     = [ s for s in arg_specs if s.kind == 'scalar_int' ]
    sflt_specs     = [ s for s in arg_specs if s.kind == 'scalar_float' ]

    # --- parameter lists ---
    # XLA FFI handler signature: Args..., Rets..., Attrs...
    param_decls = []
    bind_chain  = []
    n_arg = 0
    n_res = 0
    n_att = 0

    # Inputs: mutable inputs (will be copied to outputs), then plain arrays
    mutable_input_names = []   # arg name per buffer
    for spec in mutable_specs:
        names = []
        for buf in spec.buffers:
            name = f"arg_{ n_arg }"
            param_decls.append( f"ffi::AnyBuffer { name }" )
            bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
            names.append( name )
            n_arg += 1
        mutable_input_names.append( names )

    array_input_names = []
    for spec in array_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        array_input_names.append( name )
        n_arg += 1

    # Scalars: passed as 0-d AnyBuffer inputs (avoids Attr<T> which requires
    # jax.ffi.ffi_call kwargs support — not available in all JAX versions).
    scalar_arg_names = []
    for spec in sint_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        scalar_arg_names.append( ( name, 'int64_t' ) )
        n_arg += 1
    for spec in sflt_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        scalar_arg_names.append( ( name, 'double' ) )
        n_arg += 1

    # Outputs: mutable outputs, then return_obj, then return_tensor
    mutable_output_names = []
    for spec in mutable_specs:
        names = []
        for buf in spec.buffers:
            name = f"res_{ n_res }"
            param_decls.append( f"ffi::Result<ffi::AnyBuffer> { name }" )
            bind_chain.append( ".Ret<ffi::AnyBuffer>()" )
            names.append( name )
            n_res += 1
        mutable_output_names.append( names )

    ret_obj_names = []
    for spec in ret_obj_specs:
        names = []
        for buf in spec.buffers:
            name = f"res_{ n_res }"
            param_decls.append( f"ffi::Result<ffi::AnyBuffer> { name }" )
            bind_chain.append( ".Ret<ffi::AnyBuffer>()" )
            names.append( name )
            n_res += 1
        ret_obj_names.append( names )

    ret_tn_names = []
    for spec in ret_tn_specs:
        name = f"res_{ n_res }"
        param_decls.append( f"ffi::Result<ffi::AnyBuffer> { name }" )
        bind_chain.append( ".Ret<ffi::AnyBuffer>()" )
        ret_tn_names.append( name )
        n_res += 1

    # --- function body ---
    body = []

    # 1. Copy mutable input buffers → mutable output buffers
    for in_names, out_names in zip( mutable_input_names, mutable_output_names ):
        for a, r in zip( in_names, out_names ):
            body.append( f"    std::memcpy( { r }->untyped_data(), { a }.untyped_data(), { a }.size_bytes() );" )

    # 2. Build C++ objects for mutable args (from OUTPUT buffers)
    cpp_call_args = []
    for i, ( spec, out_names ) in enumerate( zip( mutable_specs, mutable_output_names ) ):
        views = _buf_exprs( spec, out_names, deref=True )
        expr  = _ctor_expr( spec, views )
        vname = f"cpp_mut_{ i }"
        body.append( f"    auto { vname } = { expr };" )
        cpp_call_args.append( f"{ vname }" )

    # 3. Build C++ objects for return_obj (from OUTPUT buffers)
    for i, ( spec, out_names ) in enumerate( zip( ret_obj_specs, ret_obj_names ) ):
        views = _buf_exprs( spec, out_names, deref=True )
        expr  = _ctor_expr( spec, views )
        vname = f"cpp_ret_{ i }"
        body.append( f"    auto { vname } = { expr };" )
        cpp_call_args.append( f"{ vname }" )

    # 4. Return_tensor output views
    for i, ( spec, rname ) in enumerate( zip( ret_tn_specs, ret_tn_names ) ):
        buf = spec.buffers[ 0 ]
        view = f"ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( *{ rname } )"
        vname = f"cpp_rtn_{ i }"
        body.append( f"    auto { vname } = { view };" )
        cpp_call_args.append( f"{ vname }" )

    # 5. Plain array input views
    for i, ( spec, aname ) in enumerate( zip( array_specs, array_input_names ) ):
        buf = spec.buffers[ 0 ]
        view = f"ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( { aname } )"
        vname = f"cpp_arr_{ i }"
        body.append( f"    auto { vname } = { view };" )
        cpp_call_args.append( f"{ vname }" )

    # 6. Scalar inputs — extract value from 0-d AnyBuffer
    for i, ( aname, ctype ) in enumerate( scalar_arg_names ):
        vname = f"cpp_sc_{ i }"
        body.append( f"    { ctype } { vname } = *reinterpret_cast<const { ctype }*>( { aname }.untyped_data() );" )
        cpp_call_args.append( vname )

    # 7. Call the C++ function
    body.append( f"    { func_name }( { ', '.join( cpp_call_args ) } );" )
    body.append( "    return ffi::Error::Success();" )

    # --- assemble ---
    lines = []
    sig = ", ".join( param_decls )
    lines.append( f"static ffi::Error impl_fwd_{ func_name }( { sig } ) {{" )
    lines += body
    lines.append( "}" )
    lines.append( "" )

    # XLA_FFI_DEFINE_HANDLER_SYMBOL internally calls .To(impl).release() on the binding —
    # do NOT include .To(...) here.
    bind_str = "ffi::Ffi::Bind()" + "".join( bind_chain )
    lines.append( f"XLA_FFI_DEFINE_HANDLER_SYMBOL( fwd_handler_{ func_name }, impl_fwd_{ func_name }," )
    lines.append( f"    { bind_str } );" )

    return lines


def _generate_bwd_handler( func_name: str, arg_specs: list[ FfiArgSpec ] ) -> list[ str ]:
    """Generate backward XLA FFI handler for ``func_name_backward``.

    C++ convention (matches test_alac_backward pattern):
        func_name_backward( same_fwd_cpp_args..., grad_inputs..., grad_outputs... )

    XLA FFI inputs (Arg):
        1. All output buffers of forward (mutable out + return obj + return tensor) — reconstruct same objects
        2. All plain array input buffers (saved from forward)
        3. All scalar 0-d buffers (same as forward)
        4. Grad outputs: one float buffer per float forward output

    XLA FFI outputs (Ret):
        grad inputs: one float buffer per float forward input
                     (float mutable buffers + float plain array buffers)
    """
    mutable_specs   = [ s for s in arg_specs if s.kind == 'mutable' ]
    ret_obj_specs   = [ s for s in arg_specs if s.kind == 'return_obj' ]
    ret_tn_specs    = [ s for s in arg_specs if s.kind == 'return_tensor' ]
    array_specs     = [ s for s in arg_specs if s.kind == 'array' ]
    sint_specs      = [ s for s in arg_specs if s.kind == 'scalar_int' ]
    sflt_specs      = [ s for s in arg_specs if s.kind == 'scalar_float' ]

    param_decls  = []
    bind_chain   = []
    body         = []
    n_arg = 0
    n_res = 0

    # --- Arg 1: mutable output buffers (post-forward, reconstruct same objects) ---
    mut_out_buf_names = []
    for spec in mutable_specs:
        names = []
        for buf in spec.buffers:
            name = f"arg_{ n_arg }"
            param_decls.append( f"ffi::AnyBuffer { name }" )
            bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
            names.append( name )
            n_arg += 1
        mut_out_buf_names.append( names )

    # --- Arg 2: return_obj output buffers ---
    ret_obj_buf_names = []
    for spec in ret_obj_specs:
        names = []
        for buf in spec.buffers:
            name = f"arg_{ n_arg }"
            param_decls.append( f"ffi::AnyBuffer { name }" )
            bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
            names.append( name )
            n_arg += 1
        ret_obj_buf_names.append( names )

    # --- Arg 3: return_tensor output buffers ---
    ret_tn_buf_names = []
    for spec in ret_tn_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        ret_tn_buf_names.append( name )
        n_arg += 1

    # --- Arg 4: plain array input buffers ---
    arr_buf_names = []
    for spec in array_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        arr_buf_names.append( name )
        n_arg += 1

    # --- Arg 5: scalar 0-d buffers ---
    sc_buf_names = []
    for spec in sint_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        sc_buf_names.append( ( name, 'int64_t' ) )
        n_arg += 1
    for spec in sflt_specs:
        name = f"arg_{ n_arg }"
        param_decls.append( f"ffi::AnyBuffer { name }" )
        bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
        sc_buf_names.append( ( name, 'double' ) )
        n_arg += 1

    # --- Arg 6: grad outputs (float forward outputs, same order as fwd out_shapes) ---
    # Order: float mutable out buffers, float return_obj out, float return_tensor out
    grad_out_buf_names = []
    for spec in mutable_specs:
        for buf in spec.buffers:
            if buf.dtype == 'float':
                name = f"arg_{ n_arg }"
                param_decls.append( f"ffi::AnyBuffer { name }" )
                bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
                grad_out_buf_names.append( ( name, buf ) )
                n_arg += 1
    for spec in ret_obj_specs:
        for buf in spec.buffers:
            if buf.dtype == 'float':
                name = f"arg_{ n_arg }"
                param_decls.append( f"ffi::AnyBuffer { name }" )
                bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
                grad_out_buf_names.append( ( name, buf ) )
                n_arg += 1
    for spec in ret_tn_specs:
        for buf in spec.buffers:
            if buf.dtype == 'float':
                name = f"arg_{ n_arg }"
                param_decls.append( f"ffi::AnyBuffer { name }" )
                bind_chain.append( ".Arg<ffi::AnyBuffer>()" )
                grad_out_buf_names.append( ( name, buf ) )
                n_arg += 1

    # --- Ret: grad inputs (float mutable in + float array in, same order as fwd input_arrays) ---
    grad_in_res_names = []
    for spec in mutable_specs:
        for buf in spec.buffers:
            if buf.dtype == 'float':
                name = f"res_{ n_res }"
                param_decls.append( f"ffi::Result<ffi::AnyBuffer> { name }" )
                bind_chain.append( ".Ret<ffi::AnyBuffer>()" )
                grad_in_res_names.append( ( name, buf ) )
                n_res += 1
    for spec in array_specs:
        for buf in spec.buffers:
            if buf.dtype == 'float':
                name = f"res_{ n_res }"
                param_decls.append( f"ffi::Result<ffi::AnyBuffer> { name }" )
                bind_chain.append( ".Ret<ffi::AnyBuffer>()" )
                grad_in_res_names.append( ( name, buf ) )
                n_res += 1

    # --- body: reconstruct same C++ objects as forward (from saved output buffers) ---
    cpp_call_args = []

    for i, ( spec, names ) in enumerate( zip( mutable_specs, mut_out_buf_names ) ):
        views = _buf_exprs( spec, names, deref=False )
        vname = f"cpp_mut_{ i }"
        body.append( f"    auto { vname } = { _ctor_expr( spec, views ) };" )
        cpp_call_args.append( vname )

    for i, ( spec, names ) in enumerate( zip( ret_obj_specs, ret_obj_buf_names ) ):
        views = _buf_exprs( spec, names, deref=False )
        vname = f"cpp_ret_{ i }"
        body.append( f"    auto { vname } = { _ctor_expr( spec, views ) };" )
        cpp_call_args.append( vname )

    for i, ( spec, aname ) in enumerate( zip( ret_tn_specs, ret_tn_buf_names ) ):
        buf  = spec.buffers[ 0 ]
        vname = f"cpp_rtn_{ i }"
        body.append( f"    auto { vname } = ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( { aname } );" )
        cpp_call_args.append( vname )

    for i, ( spec, aname ) in enumerate( zip( array_specs, arr_buf_names ) ):
        buf  = spec.buffers[ 0 ]
        vname = f"cpp_arr_{ i }"
        body.append( f"    auto { vname } = ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( { aname } );" )
        cpp_call_args.append( vname )

    for i, ( aname, ctype ) in enumerate( sc_buf_names ):
        vname = f"cpp_sc_{ i }"
        body.append( f"    { ctype } { vname } = *reinterpret_cast<const { ctype }*>( { aname }.untyped_data() );" )
        cpp_call_args.append( vname )

    # grad inputs (Ret) — C++ writes into these
    for i, ( rname, buf ) in enumerate( grad_in_res_names ):
        vname = f"cpp_gin_{ i }"
        body.append( f"    auto { vname } = ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( *{ rname } );" )
        cpp_call_args.append( vname )

    # grad outputs (Arg) — C++ reads these
    for i, ( aname, buf ) in enumerate( grad_out_buf_names ):
        vname = f"cpp_gout_{ i }"
        body.append( f"    auto { vname } = ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( { aname } );" )
        cpp_call_args.append( vname )

    # call backward
    body.append( f"    { func_name }_backward( { ', '.join( cpp_call_args ) } );" )
    body.append( "    return ffi::Error::Success();" )

    # --- assemble ---
    lines = []
    sig = ", ".join( param_decls )
    lines.append( f"static ffi::Error impl_bwd_{ func_name }( { sig } ) {{" )
    lines += body
    lines.append( "}" )
    lines.append( "" )

    bind_str = "ffi::Ffi::Bind()" + "".join( bind_chain )
    lines.append( f"XLA_FFI_DEFINE_HANDLER_SYMBOL( bwd_handler_{ func_name }, impl_bwd_{ func_name }," )
    lines.append( f"    { bind_str } );" )

    return lines


def _buf_exprs( spec: FfiArgSpec, names: list[ str ], deref: bool ) -> list[ str ]:
    exprs = []
    for buf, name in zip( spec.buffers, names ):
        deref_str = f"*{ name }" if deref else name
        exprs.append( f"ffi_tv<{ _dtype_cpp( buf.dtype ) },{ buf.ndim }>( { deref_str } )" )
    return exprs


def _ctor_expr( spec: FfiArgSpec, view_exprs: list[ str ] ) -> str:
    if not spec.cpp_ctor:
        return view_exprs[ 0 ]
    # cpp_ctor may contain %s placeholders
    result = spec.cpp_ctor
    for v in view_exprs:
        result = result.replace( "%s", v, 1 )
    return result


# ---------------------------------------------------------------------------
# Write + compile
# ---------------------------------------------------------------------------

def _write_and_compile( key: str, src: str ):
    from ..generated_files import generated_files
    from ..driver import driver

    device_type = driver.normalized_device_type
    ext = ".cu" if device_type == "cuda" else ".cpp"
    path = generated_files.src_dir( key ) / ( "ffi_binding" + ext )
    path.write_text( src )
    _compile( key, [ path ], device_type )
