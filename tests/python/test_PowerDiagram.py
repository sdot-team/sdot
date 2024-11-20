# from pysdot.domain_types import ScaledImage
# from pysdot import OptimalTransport
# import pylab as plt
# import numpy as np

# # helper function
# # def quantization(ot, tau=.3, niter=10):
# #     for iter in range(niter):
# #         if ot.verbosity >= 2:
# #             print( "niter quant:", iter )
# #         ot.adjust_weights()
# #         B = ot.get_centroids()
# #         ot.set_positions( ot.get_positions() + tau * ( B - ot.get_positions() ) )
# #     ot.adjust_weights()

# # initial positions
# n = 40
# positions = []
# for y in range( n ):
#     for x in range( n ):
#         positions.append( [ ( y + 0.25 + 0.5 * np.random.rand() ) / n, ( x + 0.25 + 0.5 * np.random.rand() ) / n ] )
# ot = OptimalTransport(np.array(positions))
# ot.verbosity = 2

# def f( r2, o ):
#     return ( r2**0.5 - o**2 ) ** 2
# # solve
# for n, t in enumerate( np.linspace( 1, 60, 120 ) ):
#     print( n )

#     s = 0.5 + 0.5 * np.sin( 2 * np.pi * ( t + 0 ) / 60 )
#     l = 5 * np.sin( 2 * np.pi * ( t + 40 ) / 60 )
#     o = 0.3 + 0.3 * np.sin( 2 * np.pi * ( t + 80 ) / 60 )

#     t = np.linspace(-1,1,100)
#     x, y = np.meshgrid(t,t)
#     i_0 = np.exp( -l * f( x**2 + y**2, o ) )
#     i_1 = 0.5 + ( x**2 > 0.5**2 )
    
#     img = ( 1 - s ) * i_0 + s * i_1    
#     img /= np.mean(img)
    


#     # domain
#     ot.set_domain(ScaledImage([0, 0], [1, 1], img))
#     ot.adjust_weights()

#     # display
#     nt = OptimalTransport( np.array(positions), ot.get_weights() )
#     nt.pd.display_vtk( f"results/pd_{ n }.vtk", centroids=False )




import matplotlib.pyplot as plt
from sdot import PowerDiagram
import numpy as np
# import pytest

# def text( p, txt, color ):
#     plt.text( p[ 0 ], p[ 1 ], txt, ha='center', va='center', color = color )

# pd = PowerDiagram( [ [ 0.25, 0.5 ], [ 0.75, 0.6 ], [ 0.5, 0.5 ] ] )
# pd.periodicity_transformations = [ [ 0, 1 ],  [ 1, 0 ] ]
# pd.plot()

# np.random.seed( 357 )
# positions = np.random.random( [ 40, 2 ] )

# pd = PowerDiagram( positions )
# print( pd.boundaries )
# pd.plot()

# for n in range( 60 ):
#     np.random.seed( n )
#     pd = PowerDiagram( np.random.random( [ 40, 2 ] ) 
#     print( n,  )
#     # pd.plot( plt )

best_ptp = 100000000
best_p = None
best_i = None
for i in range( 20000 ):
    np.random.seed( i )
    p = np.random.random( [ 30, 2 ] )
    pd = PowerDiagram( positions = p )

    pd.periodicity_transformations = [ [ 0, 1 ] ]

    ptp = np.ptp( pd.summary().vertex_coords[ :, 0 ] )
    if best_ptp > ptp:
        best_ptp = ptp
        best_p = p
        best_i = i

pd = PowerDiagram( best_p )
pd.periodicity_transformations = [
     [ 0, 1 ],
]
pd.plot()

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

# b0 = s.barycenters( dim = 0 )
# b1 = s.barycenters( dim = 1 )
# b2 = s.barycenters( dim = 2 )
# bc = np.array([0.5,0.5])
# print( b1 )

# pd.plot( plt )
# for lb, color in zip( [ b0, b1, b2 ], [ 'blue', 'green', 'red' ]):
#     for i, b in enumerate( lb ):
#         dir = b - bc
#         if np.linalg.norm( dir ):
#             dir = dir / np.linalg.norm( dir )
#         for e in range( 2 ):
#             if abs( dir[ e ] ) >= 0.3:
#                 dir[ e ] = dir[ e ] / abs( dir[ e ] ) # * 0.5
#         text( b + 0.03 * dir, f'{ i }', color )
# plt.xlim( -0.1, 1.1 )
# plt.ylim( -0.1, 1.1 )
# plt.show()

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
