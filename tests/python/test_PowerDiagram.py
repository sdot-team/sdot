import matplotlib.pyplot as plt
from sdot import PowerDiagram
import numpy as np
# import pytest


# pd = PowerDiagram( [ [ 0.25, 0.5 ], [ 0.75, 0.6 ], [ 0.5, 0.5 ] ] )
# pd.periodicity_transformations = [ [ 0, 1 ],  [ 1, 0 ] ]
# pd.plot()

# # np.random.seed( 357 )
# # positions = np.random.random( [ 40, 2 ] )

# # pd = PowerDiagram( positions )
# # print( pd.boundaries )
# # pd.plot()

# # for n in range( 60 ):
# #     np.random.seed( n )
# #     pd = PowerDiagram( np.random.random( [ 40, 2 ] ) 
# #     print( n,  )
# #     # pd.plot( plt )

# best_ptp = 100000000
# best_p = None
# best_i = None
# for i in range( 20000 ):
#     np.random.seed( i )
#     p = np.random.random( [ 30, 2 ] )
#     pd = PowerDiagram( positions = p )

#     pd.periodicity_transformations = [ [ 0, 1 ] ]

#     ptp = np.ptp( pd.summary().vertex_coords[ :, 0 ] )
#     if best_ptp > ptp:
#         best_ptp = ptp
#         best_p = p
#         best_i = i

# pd = PowerDiagram( best_p )
# pd.periodicity_transformations = [
#      [ 0, 1 ],
# ]
# pd.plot()

# pd.add_box_boundaries( 0, 1 )

# s = pd.summary()
# # print( s.vertex_coords )
# print( s.ref_lists[ 0 ] )
# print( s.ref_lists[ 1 ] )
# for r in range( 3 ):
#     for c in range( 3 ):
#         print( r, c )
#         for l in s.parenting[ r ][ c ]:
#             print( "  ", l )
# print( s.boundary_items )

# print( b1 )

def poute():
    from sdot import PowerDiagram
    import pylab as plt
    import numpy as np

    def text( p, txt, color ):
        plt.text( p[ 0 ], p[ 1 ], txt, ha='center', va='center', color = color )

    pd = PowerDiagram( [ [ 0.25, 0.5 ], [ 0.75, 0.6 ] ] )
    pd.add_box_boundaries( 0, 1 )
    ps = pd.summary()

    b0 = ps.barycenters( dim = 0 )
    b1 = ps.barycenters( dim = 1 )
    b2 = ps.barycenters( dim = 2 )
    bc = np.array([0.5,0.5])

    pd.plot( plt )
    for lb, color in zip( [ b0, b1, b2 ], [ 'blue', 'green', 'red' ]):
        for i, b in enumerate( lb ):
            dir = b - bc
            if np.linalg.norm( dir ):
                dir = dir / np.linalg.norm( dir )
            for e in range( 2 ):
                if abs( dir[ e ] ) >= 0.3:
                    dir[ e ] = dir[ e ] / abs( dir[ e ] ) # * 0.5
            text( b + 0.03 * dir, f'{ i }', color )
    plt.xlim( -0.1, 1.1 )
    plt.ylim( -0.1, 1.1 )
    plt.show()

# from pyvista import examples
# mesh = examples.download_st_helens()
# warped = mesh.warp_by_scalar('Elevation')
# surf = warped.extract_surface().triangulate()
# surf = surf.decimate_pro(0.75)  # reduce the density of the mesh by 75%
# surf.plot(cmap='gist_earth')

# def test_PowerDiagram():
#     # pd = PowerDiagram( positions = [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ] )

#     print( best_ptp )
#     pd = PowerDiagram( positions = best_p )

#     pd.periodicity_transformations = [
#         ( np.eye( 2 ), [ 0, 1 ] )
#     ]

#     pd.plot()

# test_PowerDiagram()

# p = PoomVec( [1, 2] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )

# cell = Cell( ndim = 3 )

# # we create a infinitely extruded triangle
# cell.cut( [ -1,  0, 0 ], 0 )
# cell.cut( [  0, -1, 0 ], 0 )
# cell.cut( [ +1, +1, 0 ], 1 )

# # of course, there's no 3D vertex...
# print( cell.nb_vertices ) # => 0

# # and this cell is sill unbounded (in the 3D space)
# print( cell.bounded ) # => False

# # ... it's because we're internally in 2D :)
# print( cell.true_dimensionality ) # => 2

# # "_td" is the shortcut suffix for "true dimensionality".
# # Methods with prefix return the information for the subspace that is defined by `cell.base`
# print( cell.nb_vertices_td ) # => 3 (the 3 vertices of the triangle)

# # we can get sample coordinates to represent these points in ndim (3D in this case)
# print( cell.vertex_coords_td @ cell.base ) # => [[0. 0. 0.] [1. 0. 0.] [0. 1. 0.]]

# # visualization will show the "sub-dimensional" content with thiner lines
# cell.plot()
# pd = PowerDiagram( [ [ 0.1, 0.1 ], [ 0.9, 0.9 ] ] )
# pd.add_box_boundaries( 0, 1 )
# pd.plot()

# # c.integral() => integration of 1 on the cell
# print( pd.cell_integrals() )

# # by default piecewise constant, positionned in [0,1]^n
# s = ScaledImage( [ [ 1, 2 ],[ 2, 1 ] ] )
# print( pd.cell_integrals( s ) )

# print( pd.cell_dintegrals_dweights( s ) )

# # [[24.20364779 -5.58586914 -1.29423728 -4.49760374 -0.72411373]
# #  [-5.58586914 10.07493089 -2.17554026  0.         -2.31352149]
# #  [-1.29423728 -2.17554026  4.35958259 -0.88980504  0.        ]
# #  [-4.49760374  0.         -0.88980504  5.38740878  0.        ]
# #  [-0.72411373 -2.31352149  0.          0.          3.03763522]]
# # max dw: 0.03202312743320352
# # [[24.27041608 -3.87760666 -2.04750292 -5.38098489 -0.82911358]
# #  [-3.87760666  9.84755188 -3.60245276  0.         -2.36749246]
# #  [-2.04750292 -3.60245276  7.36901638 -1.71906069  0.        ]
# #  [-5.38098489  0.         -1.71906069  7.10004558  0.        ]
# #  [-0.82911358 -2.36749246  0.          0.          3.19660604]]
# def pys():
#     import pysdot
#     np.random.seed( 0 )

#     ar = np.array( [ [ 1, 1 ],[ 1, 5.0 ] ] )
#     ar /= np.mean( ar )
#     do = pysdot.ScaledImage( [0,0], [1,1], ar )
#     print( do.measure() )

#     ot = pysdot.OptimalTransport( np.random.random( [ 50, 2 ] ), domain = do )
#     ot.verbosity = 3
#     ot.adjust_weights()

# def pl():
#     np.random.seed( 0 )
#     pl = optimal_transport_plan(
#         ScaledImage( [ [ 1, 1 ],[ 1, 5 ] ] ),
#         np.random.random( [ 50, 2 ] ),
#         # relaxation = 0.1
#     )
#     # [[12.10182389 -2.79293457 -0.64711864 -2.24880187 -0.36205687]
#     #  [-2.79293457  5.03746544 -1.08777013  0.         -1.15676074]
#     #  [-0.64711864 -1.08777013  2.17979129 -0.44490252  0.        ]
#     #  [-2.24880187  0.         -0.44490252  2.69370439  0.        ]
#     #  [-0.36205687 -1.15676074  0.          0.          1.51881761]]

#     # [[12.75390154 -2.14731503 -1.0620068  -2.75399311 -0.41363583]
#     #  [-2.14731503  5.06445409 -1.73506019  0.         -1.18207886]
#     #  [-1.0620068  -1.73506019  3.50263446 -0.70556747  0.        ]
#     #  [-2.75399311  0.         -0.70556747  3.45956058  0.        ]
#     #  [-0.41363583 -1.18207886  0.          0.          1.5957147 ]]

#     pl.plot()

# # pys()
# pl()
