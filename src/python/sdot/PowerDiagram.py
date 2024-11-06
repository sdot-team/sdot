from .bindings.loader import numpy_dtype_for, normalized_dtype, type_promote, module_for
import numpy as np

class PowerDiagram:
    """
        Stores all the data needed to compute a power diagram.

        Internally,
        * positions are stored in a Paving class (RegularPaving, VoronoiPaving, ...)  
        * weights are stored in a HierarchicalRepr class (, ...)

        Boundaries are represented as a 2D tensor where each row contains [ dir..., val ] where
            a point is exterior if scalar_product( dir, point ) > val.

        `dtype` is the name of the scalar type used by the bindings. One can use numpy types as input (e.g. numpy.float64).
        The static method normalized_dtype( dtype ) allows to get the internal name (e.g. for comparison).

        By default, the scalar type is deduced from the inputs (positions, weights, ...), 
        but is can be forced by specifying `dtype` in the ctor, or by setting it using `self.dtype = ...` (dtype is a property)
    """
    def __init__( self, positions = None, weights = None, boundaries = None, dtype = None, ndim = None ):
        # 
        self._binding_module = None
        self._boundaries = None
        self._positions = None
        self._weights = None
        self._dtype = None
        self._inst = None
        self._ndim = None

        # 
        self.dtype = dtype # used to force the type
        self.ndim = ndim # used to force the number of dimensions

        #
        self.boundaries = boundaries
        self.positions = positions
        self.weights = weights


    @property
    def boundaries( self ):
        return self._boundaries

    @boundaries.setter
    def boundaries( self, values ):
        if values == None:
            return
        self._boundaries = np.ascontiguousarray( values )


    @property
    def positions( self ):
        return self._positions

    @positions.setter
    def positions( self, values ):
        if values == None:
            return
        self._positions = np.ascontiguousarray( values )


    @property
    def weights( self ):
        return self._weights

    @weights.setter
    def weights( self, values ):
        if values == None:
            return
        self._weights = np.ascontiguousarray( values )


    @property
    def dtype( self ):
        # if specified by the user
        if self._dtype:
            return self._dtype
        
        # else, use the dtypes from the inputs
        dtypes = []
        if self._boundaries is not None:
            dtypes.append( normalized_dtype( self._boundaries.dtype ) )
        if self._positions is not None:
            dtypes.append( normalized_dtype( self._positions.dtype ) )
        if self._weights is not None:
            dtypes.append( normalized_dtype( self._weights.dtype ) )

        return type_promote( dtypes )

    @dtype.setter
    def dtype( self, value ):
        if value == None:
            return
        self._dtype = PowerDiagram.normalized_dtype( value )


    @property
    def ndim( self ):
        # if specified by the user
        if self._ndim:
            return self._ndim
        
        # else, use the dtypes from the inputs
        if self._boundaries is not None and len( self._boundaries ):
            return self._boundaries.shape[ 1 ]
        if self._positions is not None and len( self._positions ):
            return self._positions.shape[ 1 ]

        return None

    @ndim.setter
    def ndim( self, value ):
        if value == None:
            return
        self._ndim = int( value )

    def for_each_cell( self, function ):
        if not self._update_bindings():
            return
        
        # make `base_cell` from boundaries
        base_cell = self._binding_module.Cell()
        if self._boundaries is not None:
            ndim = self._binding_module.ndim()
            for n, bnd in enumerate( self._boundaries ):
                base_cell.cut_boundary( bnd[ : ndim ],  bnd[ ndim ], n )

        # call module function
        self._positions.for_each_cell( base_cell, self._weights, function )

    def _update_bindings( self ):
        # get the right module
        dtype = self.dtype
        ndim = self.ndim
        if dtype is None or ndim is None:
            return False

        self._binding_module = module_for( scalar_type = dtype, nb_dims = ndim )

        # check format of _positions
        if type( self._positions ) == np.ndarray:
            dv = self._binding_module.KnownVecOfPointsReader( self._positions )
            self._positions = self._binding_module.RegularGrid( dv )
        
            print( self._positions )
            raise RuntimeError("todo")
        else:
            raise "TODO"

        # check format of _weights
        if self._weights is None:
            self._weights = self._binding_module.HomogeneousWeights( 1 )
        else:
            raise "TODO"

        # check format of _boundaries
        if self._boundaries is None:
            self._boundaries = np.empty( [ 0, ndim ], dtype = numpy_dtype_for( dtype ) )
        else:
            raise "TODO"

        return True


# # pre-loaded functions
# make_weighted_point_set_aabb = vfs.function( 'make_weighted_point_set_aabb', [ f'inc_file:sdot/WeightedPointSet_AABB.h' ] )
# rt_int = vfs.function( 'Vfs::RtInt', [ 'inc_file:vfs/vfs_system/RtInt.h' ] )

# #
# class PowerDiagram:
#     def __init__( self, points_, weights_, b_dirs = None, b_offs = None ) -> None:
#         points = np.asarray( points_ )
#         weights = np.asarray( weights_ )
#         assert( points.ndim == 2 )
#         assert( weights.ndim == 1 )
#         assert( points.shape[ 0 ] == weights.shape[ 0 ] )

#         # points and weight are actually stored in a "weighted_point_set" structure
#         self.wps = make_weighted_point_set_aabb( points, weights, rt_int( points.shape[ 1 ] ) )
#         self.nb_dims = points.shape[ 1 ]

#         self.b_dirs = np.asarray( b_dirs )
#         self.b_offs = np.asarray( b_offs )
#         assert( self.b_dirs.ndim == 2 )
#         assert( self.b_offs.ndim == 1 )
#         assert( self.b_dirs.shape[ 1 ] == self.nb_dims )
#         assert( self.b_offs.shape[ 0 ] == self.b_dirs.shape[ 0 ] )

#     def write_vtk( self, filename, fit_boundaries = 0.1 ):
#         display_vtk_laguerre_cells = vfs.function( 'display_vtk_laguerre_cells', [ f'inc_file:sdot/display_vtk_laguerre_cells.h' ] )
#         VtkOutput = vfs.function( 'VtkOutput', [ f'inc_file:sdot/VtkOutput.h' ] )
#         save = vfs.method( 'save' )

#         vo = VtkOutput()
#         display_vtk_laguerre_cells( vo, self.wps, self.b_dirs, self.b_offs, fit_boundaries )
#         save( vo, filename )

#     def cell_points( self ):
#         """ return a tuple with
#              * a numpy array with the id of the diracs that made this point 
#              * a numpy array with the coordinates 
#         """
#         func = vfs.function( 'cell_points', [ f'inc_file:sdot/cell_points.h' ] )
        
#         coords, ids = func( self.wps, self.b_dirs, self.b_offs )
#         return np.reshape( coords, [ -1, self.nb_dims ] ), np.reshape( ids, [ -1, self.nb_dims + 1 ] )

