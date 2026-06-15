"""Analysis-level checks for the named batch-axis machinery (no C++ compilation).

Verifies the "raw tensor -> named batch axes" bridge ( Batched ) and the cumulation done by
BatchPlan: name-keyed axes, position resolved per object (no leading assumption), broadcast of
objects lacking an axis, and output tensors deriving axes from their named shape.
"""
import numpy
import sdot
from sdot.aggregate import Tensor, Batched, Return
from sdot.compilation.CallArgsAnalysis import CallArgsAnalysis
from sdot.compilation.BatchPlan import BatchPlan


def _arr( shape, dtype = float ):
    return sdot.driver.array( numpy.zeros( shape, dtype = dtype ) )


def test_batched_broadcast():
    # one batch axis "n" carried by max_coords only; min_coords / cut_id broadcast
    fai = CallArgsAnalysis( {
        "max_coords": Batched( _arr( ( 4, 2 ) ), [ ( "n", 0 ) ] ),
        "min_coords": Batched( _arr( ( 2, ) ),   [] ),
        "cut_id":     Batched( _arr( (), int ),  [] ),
    } )

    sub = fai.arguments.sub_dict
    assert sub[ "max_coords" ].batch_axes == [ ( "n", 0 ) ]
    assert sub[ "min_coords" ].batch_axes == []
    assert sub[ "cut_id" ].batch_axes == []

    plan = BatchPlan( fai, [ "max_coords", "min_coords", "cut_id" ] )
    assert plan.axes == [ "n" ]
    assert plan.size_exprs == [ "p.max_coords.underlying.shape( Ct<int,0>() )" ]
    assert plan.objects == [
        ( "p.max_coords", [ ( "n", 0 ) ] ),
        ( "p.min_coords", [] ),
        ( "p.cut_id", [] ),
    ]


def test_position_resolved_by_name():
    # batch axis placed *last* (trailing) on a, *first* on b: same name "n" must join, and the
    # size must be read at each object's own position (no leading assumption).
    fai = CallArgsAnalysis( {
        "a": Batched( _arr( ( 2, 5 ) ), [ ( "n", 1 ) ] ),   # core, then batch
        "b": Batched( _arr( ( 5, 2 ) ), [ ( "n", 0 ) ] ),   # batch, then core
    } )
    assert fai.arguments.sub_dict[ "a" ].batch_axes == [ ( "n", 1 ) ]

    plan = BatchPlan( fai, [ "a", "b" ] )
    assert plan.axes == [ "n" ]
    assert plan.size_exprs == [ "p.a.underlying.shape( Ct<int,1>() )" ]


def test_negative_position_normalised():
    fai = CallArgsAnalysis( { "a": Batched( _arr( ( 5, 2 ) ), [ ( "n", -2 ) ] ) } )
    assert fai.arguments.sub_dict[ "a" ].batch_axes == [ ( "n", 0 ) ]


def test_output_tensor_named_shape_axes():
    # a plain output tensor iterates over its named shape axes (here "n"), not a flat nb_items
    fai = CallArgsAnalysis( { "output": Return( Tensor( "n" ), n = 8 ) } )
    plan = BatchPlan( fai, [ "output" ] )
    assert plan.axes == [ "n" ]
    assert plan.size_exprs == [ "p.output.shape( Ct<int,0>() )" ]


if __name__ == "__main__":
    test_batched_broadcast()
    test_position_resolved_by_name()
    test_negative_position_normalised()
    test_output_tensor_named_shape_axes()
    print( "ok" )
