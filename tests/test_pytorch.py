import torch, numpy, sdot

sdot.driver.scalar_type = "float64"

def close( a, b ):
    return torch.allclose( a, torch.tensor( b, dtype = sdot.driver.dtype ) )


def check_grad( name: str, input: torch.Tensor, loss, attr, eps = 1e-4, tol = 1e-3 ):
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
    plan = sdot.plan( f, g )

    if exp_dist is not None:
        if not close( plan.distances, exp_dist ):
            raise AssertionError( f"bad distances (exp: {  exp_dist }, obt: { plan.distances })" )
    if exp_bary is not None:
        if not close( plan.barycenters, exp_bary ):
            raise AssertionError( f"bad barycenters (exp: { exp_bary }, obt: { plan.barycenters })" )

    # grad
    for proc in [ sdot.distances, sdot.barycenters ]:
        for m, attr in [ ( f, "positions" ), ( f, "weights" ), ( g, "ys" ), ( g, "xs" ) ]: #
            def loss( input ):
                setattr( m, attr, input )
                return torch.sum( proc( f, g ) )
            check_grad( name, getattr( m, attr ), loss, attr )


def test_w2_pytorch_1_dirac():
    check_plan( "0 then 1/2 => 1",
        sdot.BatchOfSumOfWeighted1dDiracs( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.Piecewise1dAffineFunction( [ 1, 1 ] ),
        [ 1 / 3, 1 / 12 ],
        [ 1 / 2, 1 / 2 ]
    )

    # non constant density
    check_plan( "0 then 1/2 => 2 x",
        sdot.BatchOfSumOfWeighted1dDiracs( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.Piecewise1dAffineFunction( [ 0, 2 ] ),
        [ 1 / 2, 1 / 12 ],
        [ 2 / 3, 2 / 3 ]
    )

    # density is normalized by default
    check_plan( "0 then 1/2 => 1 x",
        sdot.BatchOfSumOfWeighted1dDiracs( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.Piecewise1dAffineFunction( [ 0, 1 ] ),
        [ 1 / 2, 1 / 12 ],
        [ 2 / 3, 2 / 3 ]
    )

def test_w2_pytorch_2_diracs():
    check_plan( "[ 0, 1 ] => 1",
        sdot.SumOf1dWeightedDiracs( [ 0, 1 ] ),
        sdot.Piecewise1dAffineFunction( [ 1, 1 ] ),
        [ 1 / 12 ],
        [ 1 / 4, 3 / 4 ]
    )

def test_w2_pytorch_10_diracs():
    check_plan( "numpy.linspace( 0.05, 0.95, 10 ) => 1",
        sdot.SumOf1dWeightedDiracs( numpy.linspace( 0.05, 0.95, 10 ) ),
        sdot.Piecewise1dAffineFunction( [ 1, 1 ] ),
        [ 1 / 1200 ],
        numpy.linspace( 0.05, 0.95, 10 )
    )

    check_plan( "numpy.linspace( 0.05, 0.95, 10 ) => 2 x",
        sdot.SumOf1dWeightedDiracs( numpy.linspace( 0.05, 0.95, 10 ) ),
        sdot.Piecewise1dAffineFunction( [ 0, 2 ] ),
        [ 0.03405928239226341248 ]
    )
