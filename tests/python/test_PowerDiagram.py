import matplotlib.pyplot as plt
from sdot import PowerDiagram
import numpy as np
# import pytest

def text( p, txt, color ):
    plt.text( p[ 0 ], p[ 1 ], txt, ha='center', va='center', color = color )

pd = PowerDiagram( [ [ 0.25, 0.5 ], [ 0.75, 0.6 ], [ 0.5, 0.5 ] ] )
pd.periodicity_transformations = [ [ 0, 1 ],  [ 1, 0 ] ]
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
#     best_ptp = 100000000
#     best_p = None
#     for i in range( 3 ):
#         p = np.random.random( [ 30, 2 ] )
#         pd = PowerDiagram( positions = p )

#         pd.periodicity_transformations = [ [ 0, 1 ] ]

#         ptp = np.ptp( pd.summary().vertex_coords[ :, 0 ] )
#         if best_ptp > ptp:
#             best_ptp = ptp
#             best_p = p

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
