"""Default XLA-FFI protocol helpers for classes decorated with @object_with_tensors.

output_specs(cls, drv, **params)  -> list[ (name, shape, dtype) ]
from_outputs(cls, arrays, **params) -> new instance
assign_outputs(obj, arrays, **params)  -> mutate obj in-place

Classes may override these as classmethods; these functions provide the default
by introspecting TensorField descriptors.
"""
from ..object_with_tensors.TensorField import TensorField, _shape, _axis_names
from ..object_with_tensors._methods import _collect_attributes


def output_specs( cls, drv, **params ):
    """Return [(name, shape, dtype), ...] for every TensorField in cls.

    params must supply values for every axis name referenced by the fields
    (e.g. dim=2, vertex_capacity=32, ...).  Axis formulae like "dim + 1" are
    evaluated here.
    """
    if callable( getattr( cls, 'output_specs', None ) ):
        # class has its own implementation — delegate
        return cls.output_specs( drv, **params )

    specs = []
    for name, field in _collect_attributes( cls ):
        if not isinstance( field, TensorField ):
            continue
        shape = _shape_from_params( field, params )
        dtype = field.dtype if field.dtype is not None else float
        specs.append( ( name, shape, dtype ) )
    return specs


def from_outputs( cls, arrays, **params ):
    """Build a new instance of cls from a flat list of XLA output arrays."""
    if callable( getattr( cls, 'from_outputs', None ) ):
        return cls.from_outputs( arrays, **params )

    # blank object — bypass __init__
    obj = object.__new__( cls )

    fields = [ (n, f) for n, f in _collect_attributes( cls ) if isinstance( f, TensorField ) ]
    if len( arrays ) != len( fields ):
        raise RuntimeError(
            f"from_outputs: expected { len( fields ) } arrays for { cls.__name__ }, got { len( arrays ) }"
        )
    for ( name, _ ), arr in zip( fields, arrays ):
        obj.__dict__[ f'_{ name }' ] = arr

    return obj


def assign_outputs( obj, arrays, **params ):
    """Reassign TensorField attributes of *obj* from a flat list of XLA output arrays."""
    if callable( getattr( type( obj ), 'assign_outputs', None ) ):
        type( obj ).assign_outputs( obj, arrays, **params )
        return

    fields = [ (n, f) for n, f in _collect_attributes( type( obj ) ) if isinstance( f, TensorField ) ]
    if len( arrays ) != len( fields ):
        raise RuntimeError(
            f"assign_outputs: expected { len( fields ) } arrays for { type( obj ).__name__ }, got { len( arrays ) }"
        )
    for ( name, _ ), arr in zip( fields, arrays ):
        obj.__dict__[ f'_{ name }' ] = arr


# ---------------------------------------------------------------------------
def _shape_from_params( field: TensorField, params: dict ) -> list:
    """Compute the shape list for *field* using *params* instead of getattr(obj)."""
    res = []
    for axis_name in field.axis_names:
        axis_name = axis_name.replace( ' ', '' )

        if '*' in axis_name:
            raise NotImplementedError( f"_shape_from_params: '*' axis not supported: { axis_name }" )

        if '+' in axis_name:
            lhs, rhs = axis_name.split( '+' )
            res.append( params[ lhs.strip() ] + int( rhs.strip() ) )
            continue

        res.append( params[ axis_name ] )

    return res
