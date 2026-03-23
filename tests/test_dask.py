from sdot.bindings import sdot_bsp_bindings
# from matplotlib import pyplot
import dask.array as da  # type: ignore[import-untyped]
# import dask
import numpy

def avg_and_cov( points, nb_points ):
    avg = da.average( points, axis=0 )
    nrm = points - avg[ None, : ]
    cov = da.dot( nrm.T, nrm ) # TODO: remove the symetric part
    cov /= nb_points
    return da.compute( avg, cov )

def base_splits( points, max_points_per_bsp = 10, min_split = 1 ):
    """
    split points until it fits into the hardware
    """
    too_large_splits = [ {
        "nb_points": points.shape[ 0 ],
        "indices": da.arange( 0, points.shape[ 0 ], chunks = points.chunks[ 0 ] ),
        "points": points,
        "path": []
    } ]

    splits = []
    while len( too_large_splits ):
        split = too_large_splits.pop()

        split_nb_points = split[ "nb_points" ]
        split_indices = split[ "indices" ]
        split_points = split[ "points" ]
        split_path = split[ "path" ]
        if split_nb_points <= max_points_per_bsp:
            splits.append( split )
            continue

        # split_dir
        avg, cov = avg_and_cov( split_points, split_nb_points )
        eig_system = numpy.linalg.eig( cov )
        num_eigval = numpy.argmax( eig_system.eigenvalues )
        eigval = eig_system.eigenvalues[ num_eigval ]
        if eigval == 0:
            raise RuntimeError( "TODO: all the points in the same place" )

        split_dir = eig_system.eigenvectors[ num_eigval ]
        proj = split_points @ split_dir.T

        # 1D histogram
        avg_proj = numpy.dot( split_dir, avg )
        split_beg = avg_proj - 5 * eigval
        split_end = avg_proj + 5 * eigval
        hist, bins = da.compute( *da.histogram( proj, 256, range = ( split_beg, split_end ) ) )

        # split_dot
        cut_index = numpy.searchsorted( numpy.cumsum( hist ), hist.sum() / 2 )
        new_nb_points_list = [ hist[ : cut_index ].sum(), hist[ cut_index : ].sum() ]
        # if abs( new_nb_points_list[ 0 ] - new_nb_points_list[ 1 ] ) / hist.sum() >= 0.2:
        #     raise RuntimeError( "TODO: loop to find the cut for bad distributions" )
        split_dot = bins[ cut_index ]

        # filter
        new_paths = [ split_path + [ numpy.array( [ *split_dir, split_dot, inv ] ) ] for inv in range( 2 ) ]
        filter = ( split_points @ split_dir ) > split_dot

        #
        for f, new_path, new_nb_points in zip( [ filter, da.logical_not( filter ) ], new_paths, new_nb_points_list ):
            too_large_splits.append( {
                "nb_points": new_nb_points,
                "indices": split_indices[ f ],
                "points": split_points[ f ],
                "path": new_path
            } )

    return splits

points = da.random.random( ( 30, 2 ), chunks = ( 10, 2 ) ).astype( "float64" ) * da.array( [ 2, 1 ] )
splits = base_splits( points )
for split in splits:
    print( list( split[ "path" ] ), split[ "indices" ].compute() )

Bsp = sdot_bsp_bindings.Bsp_64
bsps = [ Bsp( split[ "indices" ], split[ "points" ], split[ "path" ] ) for split in splits ]

print( bsps )
# bspset = make_bsp_set(
#     [ ShapedObject( ( 10, 2 ), blk[ 0 ] ) for blk in points.to_delayed() ],
#     max_bsp_size = 15
# )

# print( bspset )
