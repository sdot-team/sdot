import dask.array as da  # type: ignore[import-untyped]
import sdot

points = da.random.random( ( 100, 2 ), chunks = ( 100, 2 ) ).astype( "float64" ) * da.array( [ 2, 1 ] )
bsp = sdot.Bsp( points, max_points_per_node = 40, max_points_per_cell = 20 )

for ind, item in enumerate( bsp.items ):
    item.compute().write_vtk( f"results/out_{ ind }.vtk" )

