from ..compilation.IoCategory import IoCategory
from ..compilation.CallArg import CallArg

class Return:
    """Declares what type a C++ function returns.

    Usage:
        driver.call( "make_hypercube", includes,
            Return( Cell, dim=2 ),
            frame, bnd
        )
        driver.call( "measure", includes,
            Return( Tensor, shape=[], dtype=float ),
            cell
        )
    The return_type must implement the protocol:
        return_type.output_specs( drv, **kwargs ) -> list[ (name, shape, dtype) ]
        return_type.from_outputs( arrays, **kwargs ) -> instance
    """
    def __init__( self, return_type, *args, **kwargs ):
        self.return_type = return_type
        self.type_kwargs = kwargs
        self.type_args   = args

    def with_prepended_batch_axis( self, N, moved_leaves, axis_name ):
        """vmap hook (see JaxDriver batch rule): prepend the batch axis `axis_name` (size `N`) to
        this output.

        A Tensor return gains a leading axis (via Tensor.make_variant); an aggregate return keeps
        its class and gains an entry in `batch_axes`. Type-stable in both cases.
        """
        kw = dict( self.type_kwargs )
        if make_variant := getattr( self.return_type, "make_variant", None ):  # Tensor return
            return Return( make_variant( [ axis_name ], 0 ), *self.type_args, **{ axis_name: N }, **kw )
        old_axes = kw.pop( "batch_axes", list( getattr( self.return_type, "batch_axes", [] ) ) )
        return Return( self.return_type, *self.type_args, batch_axes = [ axis_name ] + old_axes, **{ axis_name: N }, **kw )

    def call_arg_factory( self, call_args, parent, name_in_parent, io_category: IoCategory, ctor_args, ctor_kwargs ):
        new_io_category = IoCategory(
            want_return = True,
            want_output = True,
            has_input = False
        )
        return CallArg.factory( call_args, parent, name_in_parent, self.return_type, None, io_category = new_io_category, ctor_args = self.type_args, ctor_kwargs = self.type_kwargs )
