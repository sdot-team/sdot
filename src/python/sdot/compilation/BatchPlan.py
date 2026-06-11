from __future__ import annotations


class BatchPlan:
    """
    Resolved parallel iteration space for an `FfiCodeParallel` call.

    Built from `parallel_over` — a list of attribute paths into the `Parameters` aggregate
    (e.g. "cell", "power_diagram.cell", "output") — by asking each referenced object for its
    parallel extent. This is the single place where batch-axis cumulation lives: classes no
    longer hand-write C++ size expressions, they just name the objects to iterate over.

    The referenced object answers polymorphically:
      - an aggregate contributes its *named* `batch_axes` (sizes read from a representative
        field's leading dimensions);
      - a plain tensor contributes a single flat axis spanning all its elements
        (`<path>.nb_items()`), indexed element-wise in the body.

    Semantics of the named axes: same name ⇒ same axis ⇒ joined iteration; different names ⇒
    cartesian product. A path accumulates the axes of every level it crosses (outer + inner).

    Exposes:
      - `axes`       : the canonical ordered union of axis names (global multi-index order)
      - `size_exprs` : one C++ size expression per axis (read from a representative field's
                       leading dimension), in `axes` order
      - `objects`    : [ ( cpp_path, [ axis names of this object ] ) ], kept for the (future)
                       per-object global -> local projection inside each struct's operator().
    """

    def __init__( self, fai, parallel_over: list[ str ] ):
        self.axes: list[ str ] = []
        self.objects: list[ tuple[ str, list[ str ] ] ] = []
        self._size_expr: dict[ str, str ] = {}

        for path in parallel_over:
            call_arg = self._resolve( fai, path )
            cpp_path = "p." + path
            obj_axes, obj_sizes = self._extent_of( call_arg, cpp_path )
            self.objects.append( ( cpp_path, obj_axes ) )

            for axis, size in zip( obj_axes, obj_sizes ):
                if axis not in self._size_expr:
                    self.axes.append( axis )
                    self._size_expr[ axis ] = size

    @staticmethod
    def _extent_of( call_arg, cpp_path: str ) -> tuple[ list[ str ], list[ str ] ]:
        """Parallel axes ( names ) and their C++ sizes for one referenced object."""
        sub_dict = getattr( call_arg, "sub_dict", None )
        if sub_dict is not None:
            # aggregate: iterate over its named batch_axes. Sizes are read from a representative
            # field's leading dimensions; the index is the axis position *within this object*
            # (its own leading-dim layout), not the global union position.
            axes      = list( getattr( call_arg, "batch_axes", [] ) )
            rep_field = next( iter( sub_dict ), None )
            sizes     = [ f"{ cpp_path }.{ rep_field }.shape( Ct<int,{ i }>() )" for i in range( len( axes ) ) ]
            return axes, sizes
        # plain tensor: one flat axis over all its elements, indexed element-wise in the body.
        return [ cpp_path ], [ f"{ cpp_path }.nb_items()" ]

    @staticmethod
    def _resolve( fai, path: str ):
        """Walk the `Parameters` aggregate down `path` to the referenced CallArg_Aggregate."""
        call_arg = fai.arguments
        for part in path.split( "." ):
            call_arg = call_arg.sub_dict[ part ]
        return call_arg

    @property
    def size_exprs( self ) -> list[ str ]:
        return [ self._size_expr[ axis ] for axis in self.axes ]
