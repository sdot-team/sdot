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

num_img = 0
def plot( dirac_coords ):
    global num_img

    pyplot.plot( dirac_coords[ :, 0 ].detach().numpy(), dirac_coords[ :, 1 ].detach().numpy(), "." )
    pyplot.xlim( [ -1.1, 1.1 ] )
    pyplot.ylim( [ -1.1, 1.1 ] )
    pyplot.axis( 'equal' )

    pyplot.savefig( f"build/img_{ 1000 + num_img }", dpi=300 )
    num_img += 1
    pyplot.clf()


def loss( dirac_coords ):
    plot( dirac_coords )

    angle_list = torch.asarray( numpy.linspace( 0, numpy.pi, nb_angles, endpoint = False ), dtype = torch.float )
    proj_mat = torch.row_stack( ( torch.cos( angle_list ), torch.sin( angle_list ) ) )
    dirac_xs = proj_mat.T @ dirac_coords.T

    xs = torch.linspace( -1, 1, 30 ).repeat( ( nb_angles, 1 ) )
    ys = torch.sqrt( 1 - xs ** 2 )


    f = sdot.BatchOfPiecewise1dAffineFunctions( xs, ys )
    g = sdot.BatchOfSumOfWeighted1dDiracs( dirac_xs )
    return torch.sum( sdot.distances( f, g ) )


# solve
dirac_coords = torch.rand( ( 5, 2 ), requires_grad = True )
nb_angles = 50

sdot.helpers.solve_sgd( loss, dirac_coords )

coords = dirac_coords.detach().numpy()
print( coords )

# pyplot.plot( coords[ :, 0 ], coords[ :, 1 ], "." )
# pyplot.show()

# os.system( "convert build/img_*png build/anim.gif" )
