import inspect


class _AxisProxy:
    """Proxy passed to a condition lambda to let it resolve axis variables by attribute access."""

    def __init__( self, parent ):
        self._parent = parent

    def __getattr__( self, name ):
        if name.startswith( "_" ):
            raise AttributeError( name )
        return self._parent.value_of_axis_variable( name )


class Conditional:
    """Wraps a call argument or field descriptor, including it only when the condition holds.

    When the condition returns False the wrapped value is absent from the generated C++ struct
    and the FFI call. Each condition state compiles its own binding.

    Two usage forms:

    As a class-field annotation — condition receives an axis-variable proxy:
        @aggregate
        class Cell:
            vertex_indices : Conditional( Tensor( "nb_vertices[]", "dim+1", dtype=int ),
                                          lambda a: a.dim > 2 )

    As a driver.call() argument — condition takes no argument:
        driver.call( "f", includes,
            result_grad = Conditional( Return( Tensor, shape=[n], dtype=float ),
                                       lambda: need_grad ),
            data = my_array,
        )
    """

    def __init__( self, condition, value ):
        self.condition = condition
        self.value     = value

    def _evaluate_condition( self, parent ):
        nparams = len( inspect.signature( self.condition ).parameters )
        if nparams == 0:
            return bool( self.condition() )
        return bool( self.condition( _AxisProxy( parent ) ) )

    def call_arg_factory( self, call_args, parent, name_in_parent, io_category, ctor_args, ctor_kwargs ):
        """Value-path: self is the python_value passed to driver.call()."""
        from ..compilation.CallArg import CallArg
        if not self._evaluate_condition( parent ):
            return None
        return CallArg.factory( call_args, parent, name_in_parent, type( self.value ), self.value, io_category, ctor_args, ctor_kwargs )

    def type_call_arg_factory( self, call_args, parent, name_in_parent, python_value, io_category, ctor_args, ctor_kwargs ):
        """Type-path: self is the type annotation (descriptor) on a class field."""
        from ..compilation.CallArg import CallArg
        if not self._evaluate_condition( parent ):
            return None
        return CallArg.factory( call_args, parent, name_in_parent, self.value, python_value, io_category, ctor_args, ctor_kwargs )
