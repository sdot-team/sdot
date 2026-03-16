"""
Title: Computed Tomography Reconstruction
Tags: PyTorch, L-BFGS, Wasserstein Distance, 2D
Image: anim.gif
Description: This example demonstrates how to reconstruct a 2D density from its projections using the semi-discrete Optimal Transport (W2 distance). It uses the L-BFGS optimizer to find the optimal coordinates of Dirac points.
"""

from matplotlib import pyplot
import sdot.helpers
import numpy
import torch
import sdot
pyplot.style.use( 'dark_background' )


def loss( dirac_coords ):
    angles = torch.asarray( numpy.linspace( 0, numpy.pi, nb_angles, endpoint = False ), dtype = torch.float )
    pr_mat = torch.row_stack( ( torch.cos( angles ), torch.sin( angles ) ) )
    dir_xs = pr_mat.T @ dirac_coords.T
    dir_xs = dir_xs[ :, :, None ]
    print( dir_xs.shape )

    xs = torch.linspace( -1, 1, 30 ).repeat( ( nb_angles, 1 ) )
    ys = torch.sqrt( 1 - xs ** 2 )

    f = sdot.SumOfWeightedDiracs( dir_xs )
    g = sdot.PiecewiseAffineFunction( xs, ys )
    return torch.sum( sdot.distances( f, g ) )


def display_iter( dirac_coords, num_iter ):
    pyplot.plot( dirac_coords[ :, 0 ].detach().numpy(), dirac_coords[ :, 1 ].detach().numpy(), "*" )
    pyplot.xlim( [ -1.1, 1.1 ] )
    pyplot.ylim( [ -1.1, 1.1 ] )
    pyplot.axis( 'equal' )

    pyplot.savefig( f"build/img_{ 1000 + num_iter }", dpi=300 )
    pyplot.clf()


# solve
dirac_coords = torch.rand( ( 50, 2 ), requires_grad = True )
nb_angles = 50

sdot.helpers.solve_bfgs( loss, dirac_coords ) # , on_iter = display_iter

coords = dirac_coords.detach().numpy()
pyplot.plot( coords[ :, 0 ], coords[ :, 1 ], "*" )
pyplot.show()

# os.system( "convert build/img_*png build/anim.gif" )
