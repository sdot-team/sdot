from icecream.builtins import install
import numpy
import sdot
install()

def close( a, b, eps = 1e-5 ):
    return abs( a - b ).max() < eps

def check_grad( name: str, input, loss, attr, eps = 1e-4, tol = 1e-3 ):
    if sdot.driver.normalized_framework == "torch":
        import torch
        input.requires_grad = True
        if input.grad is not None:
            input.grad.zero_()
        c1 = loss( input )
        c1.backward()

        def with_value_at( idx, val ):
            orig = input.view( -1 )[ idx ].item()
            with torch.no_grad():
                input.view( -1 )[ idx ] = val
                res = loss( input )
                input.view( -1 )[ idx ] = orig
            return res

        for idx in range( input.numel() ):
            c0 = with_value_at( idx, input.view( -1 )[ idx ] - eps )
            c2 = with_value_at( idx, input.view( -1 )[ idx ] + eps )
            gr = ( ( c2 - c0 ) / ( 2 * eps ) ).item()
            an = input.grad.view( -1 )[ idx ].item()
            if abs( gr - an ) > tol:
                raise AssertionError( f"bad grad at idx { idx } for { attr } (ex: { gr }, an: { an }) for case '{ name }'" )


def check_plan( name: str, f, g, exp_dist = None, exp_bary = None ):
    # forward
    plan = sdot.ot_plan( f, g )

    if exp_dist is not None:
        if not close( plan.distances, sdot.driver.t1( exp_dist ) ):
            raise AssertionError( f"bad distances (exp: {  exp_dist }, obt: { plan.distances })" )
    if exp_bary is not None:
        if not close( plan.barycenters, sdot.driver.t1( exp_bary ) ):
            raise AssertionError( f"bad barycenters (exp: { exp_bary }, obt: { plan.barycenters })" )

    # grad
    if sdot.driver.normalized_dtype == "FP64":
        for proc in [ sdot.distances, sdot.barycenters ]:
            for m, attr in [ ( f, "positions" ), ( f, "weights" ), ( g, "ys" ), ( g, "xs" ) ]: #
                def loss( input ):
                    setattr( m, attr, input )
                    return proc( f, g ).sum()
                check_grad( name, getattr( m, attr ), loss, attr )

def for_each_driver_comb( cb ):
    for framework in [ "torch", "jax" ]: #
        for dtype in [ "FP32", "FP64" ]:
            for device in [ "cpu" ]: #, "cuda"
                sdot.driver.framework = framework
                sdot.driver.device = device
                sdot.driver.dtype = dtype
                cb()

def check_affine_distances():
    # constant density
    check_plan( "0 then 1/2 => 1",
        sdot.BatchOfSumOfWeightedDiracs1d( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.PiecewiseAffineGrid1d( [ 1, 1 ] ),
        [ 1 / 3, 1 / 12 ],
        [ 1 / 2, 1 / 2 ]
    )

    # non constant density
    check_plan( "0 then 1/2 => 2 x",
        sdot.BatchOfSumOfWeightedDiracs1d( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.PiecewiseAffineGrid1d( [ 0, 2 ] ),
        [ 1 / 2, 1 / 12 ],
        [ 2 / 3, 2 / 3 ]
    )

    # density is normalized by default
    check_plan( "0 then 1/2 => 1 x",
        sdot.BatchOfSumOfWeightedDiracs1d( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.PiecewiseAffineGrid1d( [ 0, 1 ] ),
        [ 1 / 2, 1 / 12 ],
        [ 2 / 3, 2 / 3 ]
    )

    #
    check_plan( "[ 0, 1 ] => 1",
        sdot.SumOfWeightedDiracs1d( [ 0, 1 ] ),
        sdot.PiecewiseAffineGrid1d( [ 1, 1 ] ),
        [ 1 / 12 ],
        [ 1 / 4, 3 / 4 ]
    )

    check_plan( "numpy.linspace( 0.05, 0.95, 10 ) => 1",
        sdot.SumOfWeightedDiracs1d( numpy.linspace( 0.05, 0.95, 10 ) ),
        sdot.PiecewiseAffineGrid1d( [ 1, 1 ] ),
        [ 1 / 1200 ],
        numpy.linspace( 0.05, 0.95, 10 )
    )

    check_plan( "numpy.linspace( 0.05, 0.95, 10 ) => 2 x",
        sdot.SumOfWeightedDiracs1d( numpy.linspace( 0.05, 0.95, 10 ) ),
        sdot.PiecewiseAffineGrid1d( [ 0, 2 ] ),
        [ 0.03405928239226341248 ]
    )

def test_piecewise_affine():
    for_each_driver_comb( check_affine_distances )


# if __name__ == "main":
# check_plan( "0 then 1/2 => 1",
#     sdot.BatchOfSumOfWeightedDiracs1d( [ [ 0 ], [ 1 / 2 ] ] ),
#     sdot.PiecewiseAffineGrid1d( [ 1, 1 ] ),
#     [ 1 / 3, 1 / 12 ],
#     [ 1 / 2, 1 / 2 ]
# )
import torch

# f = sdot.PiecewiseAffineGrid1d( [ 1, 0, 1 ] ) #
# ic( f.shape )
f = sdot.PiecewiseAffineGrid1d( [ 1, 0, 1 ] )
g = sdot.SumOfWeightedDiracs1d( [ 0, 1 ] )
p = sdot.ot_plan( f, g )

