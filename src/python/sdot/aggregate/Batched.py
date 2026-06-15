from ..compilation.CallArg_Tensor import CallArg_Tensor
from ..compilation.IoCategory import IoCategory

from ..drivers.driver import driver


class Batched:
    """Bridge for a *raw* array passed to a python method (e.g. `Cell.make_aligned_hypercubes`),
    naming which of its axes are **batch** axes.

    A raw array carries no axis names, but our batch model is name-driven (same name ⇒ joined
    iteration across objects, missing name ⇒ broadcast). `Batched` attaches that information
    explicitly, as a list of `( name, position )` pairs — the remaining axes are the per-element
    core. The position is the caller's choice (leading, trailing, ... — it may be tuned per case,
    e.g. for GPU coalescing); `BatchPlan` resolves everything by *name*, never by assuming a side.

    Negative positions are normalised against the array rank. An empty list means "no batch axis"
    (the argument is repeated / broadcast over whatever the other objects iterate).

    Example (a per-cell hypercube from min/max corners, one batch axis "n" carried by max only):
        Batched( max_coords[ N, dim ], [ ( "n", 0 ) ] )   # axis 0 is the batch axis "n"
        Batched( min_coords[ dim ],    [] )               # no batch axis -> broadcast
        Batched( cut_id[],             [] )               # scalar -> broadcast
    """

    def __init__( self, value, batch_axes: "list[ tuple[ str, int ] | str ]" = None ):
        self.value = driver.array( value )
        ndim       = self.value.ndim
        self.batch_axes = []
        for index, spec in enumerate( batch_axes or [] ):
            name, pos = spec if isinstance( spec, tuple ) else ( spec, index )
            self.batch_axes.append( ( name, pos % ndim ) )

    def call_arg_factory( self, call_args, parent, name_in_parent, io_category: IoCategory, ctor_args, ctor_kwargs ):
        return CallArg_Tensor(
            call_args, parent, name_in_parent, type( self.value ), self.value, io_category, ctor_args, ctor_kwargs,
            batch_axes = self.batch_axes,
        )
