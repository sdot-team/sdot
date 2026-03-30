# from sdot.bindings import sdot_bsp_bindings
# from matplotlib import pyplot
import dask.array as da  # type: ignore[import-untyped]
import sdot

points = da.random.random( ( 1000, 2 ), chunks = ( 100, 2 ) ).astype( "float64" ) * da.array( [ 2, 1 ] )
bsp = sdot.Bsp( points, max_points_per_node = 40 )

