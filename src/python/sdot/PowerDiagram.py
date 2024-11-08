from .bindings.loader import normalized_dtype, type_promote, sdot_module_for
from .PoomVec import PoomVec
from .Cell import Cell
import numpy as np

class PowerDiagram:
    """
        Stores all the data needed to compute a power diagram.

        Boundaries can be represented as
            * an array [ [ dir_{num_axis}, ..., val ]_{num_boundary}, ... ] where a point is exterior if scalar_product( dir, point ) > val.
            * a tuple of arrays, one for the directions (each row has a direction), and one for the values (a 1D array)

        `dtype` is the name of the scalar type used by the bindings. One can use numpy types as input (e.g. numpy.float64) or string representation (e.g. "FP32").

        By default, the scalar type and the dimension used by the bindings are deduced from the inputs (positions, weights, boundaries, ...), 
        but is can be forced by specifying `dtype` and `ndim` in the ctor, or by setting them using `self.dtype = ...` (dtype is a property) or `self.ndim = ...`

        Internally, positions and weights are stored in PoomVecs (Potientially Out of Memory vectors). They can be defined for instance using numpy arrays.

        If `automatic_conversion_to_ndarrays` is set to True, output arrays are automically converted to numpy `ndarray`.
        For instance, `self.positions` will return a `ndarray` instead of a `PoomVec`

    """
    def __init__( self, positions = None, weights = None, boundaries = None, dtype = None, ndim = None, automatic_conversion_to_ndarrays = True ):
        # base init
        self._acceleration_structure = None # paving + hierarchical repr of positions and weights
        self._binding_module = None
        self._boundaries = None # expected to be a ndarray
        self._positions = None # expected to be a ndarray
        self._weights = None # expected to be a ndarray
        self._dtype = None # can be used to force the data type
        self._ndim = None # can be used to force the dimension

        # prefs
        self.automatic_conversion_to_ndarrays = automatic_conversion_to_ndarrays

        # ctor arguments phase 1
        self.dtype = dtype # used to force the type
        self.ndim = ndim # used to force the number of dimensions

        # ctor arguments phase 2
        self.boundaries = boundaries
        self.positions = positions
        self.weights = weights


    def add_cube_boundaries( self, ndim = None, base = None, min_offset = None, max_offset = None ):
        """ delimit the power diagram by an (hyper-)cube (a square in 2D, and so on)

            if ndim is not specified, one tries to guess it from the defined attributes

            min_offset and max_offset are related to base. They can be single scalars or vector, with one value for each base item.

            If base is not specified, one takes the identity matrix
        """

        # check argument values
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

        # update the boundaries
        l = []
        if self._boundaries is not None:
            l = list( self._boundaries )
        for n in range( ndim ):
            l.append( [ -( d == n ) for d in range( ndim ) ] + [ - min_offset[ n ] ] )
            l.append( [ +( d == n ) for d in range( ndim ) ] + [ + max_offset[ n ] ] )
        self.boundaries = l

    @property
    def boundaries( self ):
        return self._boundaries

    @boundaries.setter
    def boundaries( self, values ):
        if values == None:
            return
        
        if type( values ) is tuple:
            dirs = np.asarray( values[ 0 ] )
            vals = np.asarray( values[ 1 ] )

            assert dirs.ndim == 2
            assert vals.ndim == 1
            assert dirs.shape[ 0 ] == vals.shape[ 0 ]

            self._boundaries = np.ascontiguousarray( np.concatenate( ( dirs, vals.reshape( [ -1, 1 ] ) ), axis = 1 ) )
        else:
            self._boundaries = np.ascontiguousarray( values )

    @property
    def positions( self ):
        return self._positions

    @positions.setter
    def positions( self, values ):
        if values is None:
            return
        self._positions = PoomVec( values )
        self._acceleration_structure = None

    @property
    def weights( self ):
        return self._weights

    @weights.setter
    def weights( self, values ):
        if values == None:
            return
        self._weights = PoomVec( values )
        self._acceleration_structure = None # TODO: update weight without redoing everything

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
        self._dtype = normalized_dtype( value )

    @property
    def ndim( self ):
        # if specified by the user
        if self._ndim:
            return self._ndim
        
        # else, use the ndim from the inputs
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
            Call `function( cell: Cell )` for each cell in the power diagram.

            If max_nb_threads == 1, `function` will be called from the main thread (which can be useful for instance for matplotlib, etc...)
        """
        if not self._update_acceleration_structure():
            return
        
        # make `base_cell` from boundaries
        base_cell = self._binding_module.Cell()
        if self._boundaries is not None:
            ndim = self._binding_module.ndim()
            for n, bnd in enumerate( self._boundaries ):
                base_cell.cut_boundary( bnd[ : ndim ],  bnd[ ndim ], n )

        # call module function
        cwc = lambda cell: function( Cell( _cell = cell, _binding_module = self._binding_module ) )
        self._positions.for_each_cell( base_cell, cwc, max_nb_threads or 0 )

    def plot_in_pyplot( self, fig ):
        """  
            plot cells content in pyplot figure `fig`        
        """
        self.for_each_cell( lambda cell: cell.plot_in_pyplot( fig ), max_nb_threads = 1 )

    def _update_acceleration_structure( self ):
        """ compute self._acceleration_structure if not already done


        """
        # get the module info
        dtype = self.dtype
        ndim = self.ndim
        if dtype is None or ndim is None:
            return False

        # get the compiled library
        self._binding_module = sdot_module_for( scalar_type = dtype, nb_dims = ndim )

        # early return if already done
        if self._acceleration_structure is not None and self._acceleration_structure.dtype == dtype and self._acceleration_structure.ndim == ndim:
            return

        # say that all the data are in memory
        positions = self._binding_module.KnownVecOfPointsReader( self.positions )
        weights = self._binding_module.KnownVecOfScalarsReader( self.weights )

        # make self._acceleration_structure
        self._acceleration_structure = self._binding_module.KnownVecOfPointsReader( positions, weights )

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

