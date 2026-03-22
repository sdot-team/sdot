from sdot.bindings import sdot_bsp_bindings
import dask.array as da  # type: ignore[import-untyped]
import numpy
import dask

class ShapedObject:
    def __init__( self, shape : list, value ):
        self.shape = shape
        self.value = value

def tree_reduce( list, func ):
    while len( list ) > 1:
        next_level = []
        for i in range( 0, len( list ), 2 ):
            if i + 1 < len( list ):
                next_level.append( func( list[ i ], list[ i + 1 ] ) )
            else:
                next_level.append( list[ i ] )
        list = next_level
    return list[ 0 ]

def make_bsp_set( delayed_chunks : ShapedObject, min_bsp_count = 2, max_bsp_size = 1e6 ):
    nb_points = sum( chunk.shape[ 0 ] for chunk in delayed_chunks )
    Bsp = sdot_bsp_bindings.Bsp_FP64
    dim = 2

    # start with a 1 node BSP
    too_large_bsps = [ ShapedObject( [ nb_points, dim ], dask.delayed( Bsp )( nb_points, dim ) ) ]
    bsps = []

    # while too_large_bsps
    while len( too_large_bsps ):
        bsp = too_large_bsps.pop()
        print( bsp.shape[ 0 ], max_bsp_size )
        if bsp.shape[ 0 ] <= max_bsp_size:
            bsps.append( bsp )
            continue

        # find how to cut the bsp
        avg_data_for_each_chunk = [ dask.delayed( bsp.value.avg_data_for )( chunk.value ) for chunk in delayed_chunks ]
        avg_data = tree_reduce( avg_data_for_each_chunk, dask.delayed( Bsp.avg_reduction ) )

        cov_data_for_each_chunk = [ dask.delayed( bsp.value.cov_data_for )( chunk.value, avg_data ) for chunk in delayed_chunks ]
        cov_data = tree_reduce( cov_data_for_each_chunk, dask.delayed( Bsp.cov_reduction ) )

        split_dir = Bsp.split_dir( cov_data.compute() )


        print( split_dir )
        pouet()



points = da.random.random( ( 30, 2 ), chunks = ( 10, 2 ) ).astype( "float64" )
bspset = make_bsp_set(
    [ ShapedObject( ( 10, 2 ), blk[ 0 ] ) for blk in points.to_delayed() ],
    max_bsp_size = 15
)

print( bspset )
