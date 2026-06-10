from sdot.compilation.CallArgsAnalysis import CallArgsAnalysis
from sdot.aggregate.AxisVariableSystem import AxisVariableSystem, AxisTensorSource
from sdot.aggregate.Conditional import Conditional
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
    system = AxisVariableSystem.from_sources( sources )
    assert system.local_value_of( "nb_points" ) == 5  # first value found
    try:
        system.check_consistency()
        assert False, "expected an inconsistency error"
    except ValueError:
        pass


def test_conditional_true_includes_field():
    """A Conditional whose lambda returns True behaves as if the value were given directly."""
    t = numpy.zeros( 3 )
    c_true  = CallArgsAnalysis( { "x": Conditional( lambda: True, t ) } )
    c_plain = CallArgsAnalysis( { "x": t } )
    assert "x" in c_true.arguments.sub_dict
    assert c_true.arguments.signature() == c_plain.arguments.signature()


def test_conditional_false_removes_field():
    """A Conditional whose lambda returns False removes the field from sub_dict and the FFI lists."""
    t = numpy.zeros( 3 )
    c_false = CallArgsAnalysis( { "x": Conditional( lambda: False, t ) } )
    assert "x" not in c_false.arguments.sub_dict
    assert len( c_false.non_differentiable_tensor_inputs ) == 0
    assert len( c_false.differentiable_tensor_inputs ) == 0


def test_conditional_signatures_differ():
    """True and False states must produce different binding signatures."""
    t = numpy.zeros( 3 )
    sig_true  = CallArgsAnalysis( { "x": Conditional( lambda: True , t ) } ).arguments.signature()
    sig_false = CallArgsAnalysis( { "x": Conditional( lambda: False, t ) } ).arguments.signature()
    assert sig_true != sig_false


def test_conditional_false_absent_token_in_signature():
    """The absent token 'x_absent' must appear in the False-state signature."""
    t = numpy.zeros( 3 )
    sig = CallArgsAnalysis( { "x": Conditional( lambda: False, t ) } ).arguments.signature()
    assert "x_absent" in sig


if __name__ == "__main__":
    test_pin_from_shape()
    test_several_tensors_share_axes()
    test_descend_into_children()
    test_ascend_to_prefixed_kwargs()
    test_inconsistent_shapes_are_rejected()
    test_conditional_true_includes_field()
    test_conditional_false_removes_field()
    test_conditional_signatures_differ()
    test_conditional_false_absent_token_in_signature()
    print( "ok" )
