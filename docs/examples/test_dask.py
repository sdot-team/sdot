# from sdot.bindings import sdot_bsp_bindings
# from matplotlib import pyplot
import dask.array as da  # type: ignore[import-untyped]
from icecream import ic
import sdot

points = da.random.random( ( 10, 2 ), chunks = ( 10, 2 ) ).astype( "float64" ) * da.array( [ 2, 1 ] )
bsp = sdot.Bsp( points, max_points_per_bsp = 9 )

for i, item in enumerate( bsp.items ):
    # item.write_vtk( f"build/out_{ i }.vtk" )
    ic( item.compute() )
