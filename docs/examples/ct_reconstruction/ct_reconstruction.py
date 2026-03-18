"""
Title: Computed Tomography Reconstruction
Tags: PyTorch, L-BFGS, Wasserstein Distance, 2D
Image: anim.gif
Description: This example demonstrates how to reconstruct a 2D density from its projections using the semi-discrete Optimal Transport (W2 distance). It uses the L-BFGS optimizer to find the optimal coordinates of Dirac points.
"""

from matplotlib import pyplot
import torch
import sdot

class Sinogram:
    def __init__( self, nb_angles = 1000, nb_pixels = 1000 ):
        self.angles = torch.linspace( 0, torch.pi, nb_angles + 1, dtype = torch.float )[ : -1 ]
        self.xs = torch.linspace( -1, 1, nb_pixels )[ None, : ].repeat( nb_angles, 1 )
        self.ys = torch.zeros( [ nb_angles, nb_pixels ] )

    def add_disk( self, center, radius, density ):
        # TODO: make the integrals
        center = torch.tensor( center )
        angles = self.angles[ :, None ].repeat( ( 1, self.ys.shape[ 1 ] ) )
        self.ys += density * torch.sqrt( torch.max( torch.tensor( 0 ), radius ** 2 - ( self.xs - center[ 0 ] * torch.cos( angles ) - center[ 1 ] * torch.sin( angles ) ) ** 2 ) )

    def projection( self, dirac_coords ):
        proj_mat = torch.row_stack( ( torch.cos( sinogram.angles ), torch.sin( sinogram.angles ) ) )
        return proj_mat.T @ dirac_coords.T


num_img = 0
def plot( dirac_coords, max_num = 10000 ):
    global num_img

    if num_img < max_num:
        pyplot.plot( dirac_coords[ :, 0 ].detach().numpy(), dirac_coords[ :, 1 ].detach().numpy(), "." )
        pyplot.xlim( [ -1.1, 1.1 ] )
        pyplot.ylim( [ -1.1, 1.1 ] )
        pyplot.axis( 'equal' )

        pyplot.savefig( f"build/img_{ 1000 + num_img }", dpi=100 )
        pyplot.clf()

    num_img += 1


def loss( dirac_coords ):
    dirac_xs = sinogram.projection( dirac_coords )

    f = sdot.BatchOfPiecewise1dAffineFunctions( sinogram.xs, sinogram.ys )
    g = sdot.BatchOfSumOfWeighted1dDiracs( dirac_xs )
    print( torch.sum( sdot.distances( f, g ) ) )
    return torch.sum( sdot.distances( f, g ) )

sinogram = Sinogram( 1000, 1000 )
sinogram.add_disk( [ 0.0, 0.0 ], 0.50, + 1.0 )
sinogram.add_disk( [ 0.0, 0.0 ], 0.45, - 1.0 )
sinogram.add_disk( [ 0.2, 0.0 ], 0.10, + 1.0 )

dirac_coords = torch.rand( ( 1000, 2 ), requires_grad = True )
sdot.driver.optimize_using_lbfgs( loss, dirac_coords )
# sdot.driver.optimize_using_sgd( loss, dirac_coords )

coords = dirac_coords.detach().numpy()

# pyplot.style.use( 'dark_background' )
pyplot.plot( coords[ :, 0 ], coords[ :, 1 ], "." )
pyplot.axis( 'equal' )
pyplot.show()


# import os
# os.system( "rm build/img_*png" )
# os.system( "convert -delay 5 build/img_*png build/anim.gif" )
