from .bindings.loader import numpy_dtype_for, normalized_dtype, type_promote, module_for
from .Cell import Cell
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


    def add_cube_boundaries( self, ndim = None, base = None, min_offset = None, max_offset = None ):
        """ add an (hyper-)cube to the boundaries

            if ndim is not specified, one tries to guess it from the defined attributes

            min_offset and max_offset are related to base. They can be single scalars or vector, with one value for each base item. 
        """
        if base is not None:
            ndim = len( base )
        if ndim is None:
            ndim = self.ndim
            if ndim is None:
                raise RuntimeError( "Found no way to guess ndim" )
        if base is None:
            base = np.identity( ndim )
        if min_offset is None:
            min_offset = 0
        if max_offset is None:
            max_offset = 1
        if isinstance( min_offset, ( int, float ) ):
            min_offset = np.full( [ ndim ], min_offset )
        if isinstance( max_offset, ( int, float ) ):
            max_offset = np.full( [ ndim ], max_offset )

        l = []
        if self._boundaries is not None:
            l = list( self._boundaries )
        for n in range( ndim ):
            l.append( [ -( d == n ) for d in range( ndim ) ] + [ - min_offset[ n ] ] )
            l.append( [ +( d == n ) for d in range( ndim ) ] + [ + max_offset[ n ] ] )
        self._boundaries = np.array( l )

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
        if values is None:
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
            return self._boundaries.shape[ 1 ] - 1
        if self._positions is not None and len( self._positions ):
            return self._positions.shape[ 1 ]

        return None

    @ndim.setter
    def ndim( self, value ):
        if value == None:
            return
        self._ndim = int( value )

    def for_each_cell( self, function, max_nb_threads = None ):
        """
        """
        if not self._update_bindings():
            return
        
        # make `base_cell` from boundaries
        base_cell = self._binding_module.Cell()
        if self._boundaries is not None:
            ndim = self._binding_module.ndim()
            for n, bnd in enumerate( self._boundaries ):
                base_cell.cut_boundary( bnd[ : ndim ],  bnd[ ndim ], n )

        # call module function
        cwc = lambda cell: function( Cell( _cell = cell, _binding_module = self._binding_module ) )
        self._positions.for_each_cell( base_cell, self._weights, cwc, max_nb_threads or 0 )

    def plot_in_pyplot( self, fig ):
        """  
            plot cells content in pyplot figure `fig`        
        """
        self.for_each_cell( lambda cell: cell.plot_in_pyplot( fig ), max_nb_threads = 1 )

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
        else:
            print( type( self._positions ) )
            raise "TODO"

        # check format of _weights
        if self._weights is None:
            self._weights = self._binding_module.LocalWeightBounds_ConstantValue( 1 )
        else:
            raise "TODO"

        # check format of _boundaries
        if self._boundaries is None:
            self._boundaries = np.empty( [ 0, ndim ], dtype = numpy_dtype_for( dtype ) )
        elif not isinstance( self._boundaries, np.ndarray ):
            raise RuntimeError( "boundaries must be expressed as a ndarray" )

        return True


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

