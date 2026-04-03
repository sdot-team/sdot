from icecream.builtins import install
import numpy
import sdot
install()

def close( a, b, eps = 1e-5 ):
    return abs( a - b ).max() < eps

def check_plan( name: str, f, g, exp_dist = None, exp_bary = None ):
    # forward
    plan = sdot.ot_plan( f, g )

    if exp_dist is not None:
        if not close( plan.distances, sdot.driver.t1( exp_dist ) ):
            raise AssertionError( f"bad distances (exp: {  exp_dist }, obt: { plan.distances }) for case '{ name }'" )
    if exp_bary is not None:
        if not close( plan.barycenters, sdot.driver.t1( exp_bary ) ):
            raise AssertionError( f"bad barycenters (exp: { exp_bary }, obt: { plan.barycenters }) for case '{ name }'" )

    # for the backward test we need batch version, so that empty tensor => None (convention
    bf = f if isinstance( f, sdot.BatchOfDistributions ) else f.batch_version( 1 )
    bg = g if isinstance( g, sdot.BatchOfDistributions ) else g.batch_version( 1 )

    #
    _flat_inputs = bf.flat_tensor_list() + bg.flat_tensor_list()

    # a function to call ot_plan from a flat list of tensors
    _apply_fn = sdot.make_apply_fn( bf, bg )

    # only dirac pos
    flat_inputs = _flat_inputs[ :2 ]
    def apply_fn( *x ):
        return _apply_fn( *x, *_flat_inputs[ 2: ] )

    # backward: check all gradients at once
    if sdot.driver.normalized_framework == "torch" and sdot.driver.normalized_dtype == "FP64":
        import torch
        flat_inputs = [ t.detach().requires_grad_( t.ndim > 0 ) for t in flat_inputs ]
        try:
            torch.autograd.gradcheck( apply_fn, flat_inputs, eps=1e-4, atol=1e-3, raise_exception=True )
        except Exception as e:
            raise AssertionError( f"bad grad in case '{ name }': { e }" )

    if sdot.driver.normalized_framework == "jax" and sdot.driver.normalized_dtype == "FP64":
        from jax.test_util import check_grads
        try:
            check_grads( apply_fn, flat_inputs, order=1, modes=[ 'rev' ], eps=1e-4, atol=1e-3 )
        except Exception as e:
            raise AssertionError( f"bad grad in case '{ name }': { e }" )


def for_each_driver_comb( cb ):
    for framework in [ "torch" ]: #, "jax"
        for dtype in [ "FP64", "FP32" ]: #
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
        sdot.PiecewiseAffineGrid1d( [ 0, 2 ], [ 0, 1 ] ),
        [ 1 / 2, 1 / 12 ],
        [ 2 / 3, 2 / 3 ]
    )

    # density is normalized by default
    check_plan( "0 then 1/2 => 1 x",
        sdot.BatchOfSumOfWeightedDiracs1d( [ [ 0 ], [ 1 / 2 ] ] ),
        sdot.PiecewiseAffineGrid1d( [ 0, 1 ], [ 0, 1 ] ),
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

    check_plan( "numpy.linspace( 0.05, 0.95, 10 ) => several knots",
        sdot.SumOfWeightedDiracs1d( numpy.linspace( 0.05, 0.95, 10 ) ),
        sdot.PiecewiseAffineGrid1d( [ 1, 0, 2 ], knots = [ 0, 1, 2 ] ),
    )

def test_piecewise_affine():
    for_each_driver_comb( check_affine_distances )

import torch

check_plan( "[ 0, 1 ] => 1",
    sdot.SumOfWeightedDiracs1d( [ 0, .5, 1, 2 ], [ 1, 1, 1, 2 ] ),
    sdot.PiecewiseAffineGrid1d( [ 1, 2, 1 ] ),
)

# check_plan( "numpy.linspace( 0.05, 0.95, 10 ) => 1",
#     sdot.SumOfWeightedDiracs1d( numpy.linspace( 0.05, 0.95, 10 ) ),
#     sdot.PiecewiseAffineGrid1d( [ 1, 1 ] ),
#     [ 1 / 1200 ],
#     numpy.linspace( 0.05, 0.95, 10 )
# )
