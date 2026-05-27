from sdot.compilation.CallArgsAnalysis import CallArgsAnalysis
from sdot.aggregate.AxisVariableSystem import AxisVariableSystem, AxisTensorSource
from sdot.aggregate.AxisExpr import AxisExpr
from sdot.aggregate.Tensor import Tensor
from sdot.aggregate import aggregate

import numpy


def test_pin_from_shape():
    """A defined tensor pins its axis variables through its shape (offset included)."""
    @aggregate
    class Test:
        a : Tensor( "dim + 1" )

    c = CallArgsAnalysis( { "test": Test( a = [ 1, 2 ] ) } )
    assert c.arguments.value_of_axis_variable( "dim" ) == 1


def test_several_tensors_share_axes():
    """Axis variables shared by several tensors of the same aggregate must agree."""
    @aggregate
    class Dist:
        positions : Tensor( "nb_points", "dim" )
        weights   : Tensor( "nb_points" )

    d = Dist( positions = numpy.zeros( ( 5, 3 ) ), weights = numpy.zeros( 5 ) )
    assert d.nb_points == 5
    assert d.dim == 3


def test_descend_into_children():
    """An aggregate resolves an axis variable by searching the tensors of its children."""
    @aggregate
    class Inner:
        positions : Tensor( "nb_points", "dim" )

    @aggregate
    class Outer:
        inner : Inner

    o = Outer( inner = Inner( positions = numpy.zeros( ( 4, 2 ) ) ) )
    top = CallArgsAnalysis( { "o": o } ).arguments
    assert top.value_of_axis_variable( "dim" ) == 2
    assert top.value_of_axis_variable( "nb_points" ) == 4


def test_ascend_to_prefixed_kwargs():
    """An undefined (output) tensor gets its axis from an explicit, path-prefixed kwarg of a parent."""
    @aggregate
    class Inner:
        out : Tensor( "dim" )

    @aggregate
    class Outer:
        inner : Inner

    c = CallArgsAnalysis( { "o": Outer( inner = Inner( out = None ) ) } )
    c.arguments.ctor_kwargs = { "o_inner_out_dim": 7 }

    out = c.arguments.sub_dict[ "o" ].sub_dict[ "inner" ].sub_dict[ "out" ]
    assert out.value_of_axis_variable( "dim" ) == 7


def test_inconsistent_shapes_are_rejected():
    """check_consistency catches two shapes implying different values for the same axis."""
    sources = [
        AxisTensorSource( shape = [ AxisExpr( "nb_points" ) ], numpy_value = numpy.zeros( 5 ) ),
        AxisTensorSource( shape = [ AxisExpr( "nb_points" ) ], numpy_value = numpy.zeros( 7 ) ),
    ]
    system = AxisVariableSystem( sources )
    assert system.local_value_of( "nb_points" ) == 5  # first value found
    try:
        system.check_consistency()
        assert False, "expected an inconsistency error"
    except ValueError:
        pass


if __name__ == "__main__":
    test_pin_from_shape()
    test_several_tensors_share_axes()
    test_descend_into_children()
    test_ascend_to_prefixed_kwargs()
    test_inconsistent_shapes_are_rejected()
    print( "ok" )
