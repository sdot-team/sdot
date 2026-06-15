from __future__ import annotations


class BatchPlan:
    """
    Resolved parallel iteration space for an `FfiCodeParallel` call.

    Built from `parallel_over` — a list of attribute paths into the `Parameters` aggregate
    (e.g. "cell", "power_diagram.cell", "output") — by asking each referenced object for its
    parallel axes. This is the single place where batch-axis cumulation lives: classes no longer
    hand-write C++ size expressions, they just name the objects to iterate over.

    Everything is keyed by *axis name* (same name ⇒ same axis ⇒ joined iteration; different names
    ⇒ cartesian product; an object missing a name ⇒ broadcast). Crucially, each object reports the
    *position* of each of its axes, so nothing assumes batch axes sit at the leading positions —
    the side (leading / trailing) is free and may be tuned per case (e.g. GPU coalescing).

    Each referenced object answers polymorphically:
      - an aggregate contributes its named `batch_axes` (sizes read from a representative field's
        dimensions, at the matching positions);
      - a tensor declared via `Batched` contributes exactly its declared `( name, position )` axes;
      - a plain output tensor contributes all its named shape axes (one parallel task per element);
      - a plain *input* tensor is rejected (it must be wrapped in `Batched` to name its axes).

    Exposes:
      - `axes`       : the canonical ordered union of axis names (global multi-index order)
      - `size_exprs` : one C++ size expression per axis, in `axes` order
      - `objects`    : [ ( cpp_path, [ ( axis_name, position ) ] ) ], for the per-object
                       global -> local projection inside each struct's operator().
    """

    def __init__( self, fai, parallel_over: list[ str ] ):
        self.axes: list[ str ] = []
        self.objects: list[ tuple[ str, list[ tuple[ str, int ] ] ] ] = []
        self._size_expr: dict[ str, str ] = {}

        for path in parallel_over:
            call_arg = self._resolve( fai, path )
            cpp_path = "p." + path
            axes, size_of = self._extent_of( call_arg, cpp_path )
            self.objects.append( ( cpp_path, axes ) )

            for name, _position in axes:
                if name not in self._size_expr:
                    self.axes.append( name )
                    self._size_expr[ name ] = size_of[ name ]

    @staticmethod
    def _resolve( fai, path: str ):
        """Walk the `Parameters` aggregate down `path` to the referenced CallArg."""
        call_arg = fai.arguments
        for part in path.split( "." ):
            call_arg = call_arg.sub_dict[ part ]
        return call_arg

    @staticmethod
    def _extent_of( call_arg, cpp_path: str ):
        """Return ( [ ( name, position ) ], { name: cpp_size_expr } ) for one referenced object."""
        # A *raw tensor* named via `Batched` is wrapped in a `BatchOf<…>` at the Parameters root, so
        # its shapes are read through `.underlying`. An *aggregate* is not wrapped ( it keeps its own
        # named operator() ), nor is a plain output tensor ( axes from its shape ); both read direct.
        sub_dict = getattr( call_arg, "sub_dict", None )
        if sub_dict is not None:
            # aggregate: its named batch_axes are leading dimensions of every field; read sizes
            # from a representative field at the matching position.
            names     = list( getattr( call_arg, "batch_axes", [] ) )
            rep_field = next( iter( sub_dict ), None )
            axes      = [ ( name, i ) for i, name in enumerate( names ) ]
            size_of   = { name: f"{ cpp_path }.{ rep_field }.shape( Ct<int,{ i }>() )" for name, i in axes }
            return axes, size_of

        # tensor
        declared = getattr( call_arg, "batch_axes", None )
        if declared is not None:                       # Batched: exactly the declared axes ( wrapped )
            axes = list( declared )
            size_of = { name: f"{ cpp_path }.underlying.shape( Ct<int,{ pos }>() )" for name, pos in axes }
        elif call_arg.io_category.want_output:         # output: every named shape axis is parallel
            axes = BatchPlan._output_axes( call_arg, cpp_path )
            size_of = { name: f"{ cpp_path }.shape( Ct<int,{ pos }>() )" for name, pos in axes }
        else:
            raise RuntimeError(
                f"{ cpp_path }: a raw input tensor in `parallel_over` must be wrapped in Batched(...) "
                f"to name its batch axes" )

        return axes, size_of

    @staticmethod
    def _output_axes( call_arg, cpp_path: str ) -> list[ tuple[ str, int ] ]:
        """Named `( name, position )` axes of an output tensor, one per declared shape axis."""
        axes = []
        for position, expr in enumerate( call_arg.shape ):
            names: list[ str ] = []
            expr.get_axis_variable_names( names )
            name = names[ 0 ] if len( names ) == 1 else f"{ cpp_path }_{ position }"
            axes.append( ( name, position ) )
        return axes

    @property
    def size_exprs( self ) -> list[ str ]:
        return [ self._size_expr[ name ] for name in self.axes ]
