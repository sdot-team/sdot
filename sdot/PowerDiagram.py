from .bindings.loader import normalized_dtype, numpy_dtype_for, type_promote, module_for
from .distributions.normalized_distribution import normalized_distribution
from .bindings.integration_module import integration_module
from .TransformationMatrix import TransformationMatrix
from .PowerDiagramSummary import PowerDiagramSummary
from .VtkOutput import VtkOutput
from .PoomVec import PoomVec
from .Expr import Expr
from .Cell import Cell

from munch import Munch
import numpy as np



class PowerDiagram:
    """
        Stores all the data needed to compute a power diagram, i.e.
            * seed positions
            * seed weights
            * optional boundaries
            * an optional underlying measure and an optional underlying radius function for method like integration, barycenter, ...

        Boundaries can be represented as
            * an array [ [ dir_{num_axis}, ..., val ]_{num_boundary}, ... ] where a point `x` is considered as exterior if `np.dot( dir, point ) > val`.
            * a tuple of arrays, one for the directions (each row has a direction), and one for the values (a 1D array)

        Underlying measure is internally stored as an instance of `Distribution`. It is used to compute measures (volumes/areas/...) and can give boundaries (for instance, IndicatorFunction(UnitBox()) will naturally add boundaries).

        `dtype` is the name of the scalar type used by the bindings. One can use numpy types as input (e.g. numpy.float64) or string representation (e.g. "FP32").

        By default, the scalar type and the dimension used by the bindings are deduced from the inputs (positions, weights, boundaries, ...), 
        but is can be forced by specifying `dtype` and `ndim` in the ctor, or by setting them using `self.dtype = ...` (dtype is a property) or `self.ndim = ...`

        Internally, positions and weights are stored in PoomVecs (Potientially Out of Memory vectors). They can be defined for instance using numpy arrays.

        If `automatic_conversion_to_ndarrays` is set to True, output arrays are automically converted to numpy `ndarray`.
        For instance, `self.positions` will return a `ndarray` instead of a `PoomVec`

    """
    def __init__( self, positions = None, weights = None, boundaries = None, underlying_measure = None, dtype = None, ndim = None, automatic_conversion_to_ndarrays = True ):
        # base init
        self._density_under_final_boundaries = None # symbolic expression representing the value inside the convex boundaries 
        self._periodicity_transformations = [] # list of TransformationMatrix 
        self._acceleration_structure = None # paving + hierarchical repr of positions and weights
        self._underlying_measure = None # instance of Distribution
        self._binding_module = None
        self._boundaries = None # expected to be a ndarray
        self._positions = None # PoomVec
        self._weights = None # PoomVec
        self._dtype = None # can be used to force the data type
        self._ndim = None # can be used to force the dimension

        # prefs
        self.automatic_conversion_to_ndarrays = automatic_conversion_to_ndarrays

        # ctor arguments phase 1
        self.dtype = dtype # used to force the type
        self.ndim = ndim # used to force the number of dimensions

        # ctor arguments phase 2
        self.underlying_measure = underlying_measure
        self.boundaries = boundaries
        self.positions = positions
        self.weights = weights

    def add_box_boundaries( self, min_offset = None, max_offset = None, ndim = None, base = None ):
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
    def periodicity_transformations( self ):
        return self._periodicity_transformations
    
    @periodicity_transformations.setter
    def periodicity_transformations( self, values ):
        self._periodicity_transformations = [ TransformationMatrix( v ) for v in values ]
        self._acceleration_structure = None

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
        if self.automatic_conversion_to_ndarrays and self._positions is not None:
            return self._positions.as_ndarray
        return self._positions

    @positions.setter
    def positions( self, values ):
        if values is None:
            return
        self._acceleration_structure = None
        self._positions = PoomVec( values )

    @property
    def weights( self ):
        if self.automatic_conversion_to_ndarrays and self._weights is not None:
            return self._weights.as_ndarray
        return self._weights

    @weights.setter
    def weights( self, values ):
        if values is None:
            return
        self._weights = PoomVec( values )
        self._weight_have_been_modified()

    def _weight_have_been_modified( self ):
        self._acceleration_structure = None # TODO: update weight without redoing everything

    @property
    def underlying_measure( self ):
        return self._underlying_measure

    @underlying_measure.setter
    def underlying_measure( self, values ):
        self._underlying_measure = normalized_distribution( values, "Lebesgue" )

    @property
    def dtype( self ):
        # if specified by the user
        if self._dtype:
            return self._dtype
        
        # else, use the dtypes from the inputs
        dtypes = []
        for v in [ self._boundaries, self._positions, self._weights ]:
            if v is not None:
                dtypes.append( normalized_dtype( v.dtype ) )

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
        if self._positions is not None:
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

        if not self._update_internal_attributes():
            return

        # call the compiled function and wrap the cell
        cwc = lambda cell: function( Cell( _cell = cell, _binding_module = self._binding_module ) )
        self._acceleration_structure.for_each_cell( self._base_cell._cell, cwc, max_nb_threads or 0 )

    def cell_integrals( self, func = 1 ):
        """ get integration of `func` other each cell """
        if not self._update_internal_attributes():
            return np.empty( [ 0, self.ndim or 0 ] )
        
        # very naughty...
        from .distributions.ScaledImage import ScaledImage

        if func == 1 and isinstance( self._underlying_measure, ScaledImage ):
            import pysdot 
            mi = [ 0 for _ in range( self.ndim ) ]
            ma = [ 1 for _ in range( self.ndim ) ]
            ar = self._underlying_measure.array / np.mean( self._underlying_measure.array )
            pd = pysdot.PowerDiagram( self.positions, self.weights, domain = pysdot.ScaledImage( mi, ma, ar ) )
            return pd.integrals()

        if isinstance( func, ScaledImage ):
            import pysdot 
            mi = [ 0 for _ in range( self.ndim ) ]
            ma = [ 1 for _ in range( self.ndim ) ]
            ar = func.array / np.mean( func.array )
            pd = pysdot.PowerDiagram( self.positions, self.weights, domain = pysdot.ScaledImage( mi, ma, ar ) )
            return pd.integrals()

        # final expression to integrate other the cell
        final_expr = self._density_under_final_boundaries * Expr( func )

        # not so naughty...
        cv = final_expr.constant_value()
        if cv is not None:
            return self._binding_module.measures( self._acceleration_structure, self._base_cell._cell, cv )

        module, rt_data = integration_module( final_expr, self.dtype, self.ndim )
        return module.power_diagram_integrals( self._acceleration_structure, self._base_cell._cell, rt_data )

    def cell_dintegrals_dweights( self, func = 1 ):
        """ 
            get derivatives of integration of `func` other each cell

            return ( m_rows, m_cols, m_vals, v_vals, error_code ) with m = matrix, v = vector
        """
        if not self._update_internal_attributes():
            raise ( [], [], [], [], 0 )

        # very naughty...
        from .distributions.ScaledImage import ScaledImage
        if func == 1 and isinstance( self._underlying_measure, ScaledImage ):
            import pysdot 
            mi = [ 0 for _ in range( self.ndim ) ]
            ma = [ 1 for _ in range( self.ndim ) ]
            ar = self._underlying_measure.array / np.mean( self._underlying_measure.array )
            pd = pysdot.PowerDiagram( self.positions, self.weights, domain = pysdot.ScaledImage( mi, ma, ar ) )
            mvs = pd.der_integrals_wrt_weights()
            # print( pd.integrals() )
            from scipy.sparse import csr_matrix
            M = csr_matrix( ( mvs.m_values, mvs.m_columns, mvs.m_offsets ) )
            C = M.tocoo()
            # print( M.todense() )
            # print( C.data )
            # print( "v", mvs.m_values )
            # todo()
            return C.coords[ 0 ].copy(), C.coords[ 1 ].copy(), C.data.copy(), mvs.v_values.copy(), 0

        # final expression to integrate other the cell
        final_expr = self._density_under_final_boundaries * Expr( func )

        cv = final_expr.constant_value()
        if cv is not None:
            return self._binding_module.dmeasures_dweights( self._acceleration_structure, self._base_cell._cell, cv )

        raise RuntimeError( "TODO" )
    
    def summary( self ):
        """ return a PowerDiagramSummary with arrays that contain global information to summarize the power diagram (coords, parenting, ...).
        """
        if not self._update_internal_attributes():
            raise ValueError( "missing some attribute to be able to make a summary" )

        # TODO: PowerDiagram in sub-spaces
        s = self._binding_module.summary( self._acceleration_structure, self._base_cell._cell )
        return PowerDiagramSummary( *s, np.eye( self.ndim ) )

    def plot_pyplot( self, fig, **kwargs ):
        """  
            For optional arguments, see the `Cell.plot` method
        """
        if fig is None:
            import matplotlib.pyplot as plt
            self.plot( plt, **kwargs )
            plt.show()
            return

        if self.ndim == 3:
            fig = fig.figure()
            fig = fig.add_subplot( projection = '3d' )

        self.for_each_cell( lambda cell: cell.plot( fig, **kwargs ), max_nb_threads = 1 )
        return fig

    def plot_vtk( self, vtk_output, **kwargs ):
        if isinstance( vtk_output, str ):
            vo = VtkOutput()
            self.plot_vtk( vo, **kwargs )
            vo.save( vtk_output )
            return 
        
        assert isinstance( vtk_output, VtkOutput )

        self._update_internal_attributes()
        self._binding_module.plot_vtk( vtk_output._inst, self._acceleration_structure, self._base_cell._cell )

    def plot( self, fig = None, **kwargs ):
        """  
            plot cells content in
            * pyplot figure (`pyplot` or in a `pyplot.fig` for instance)
            * or in a VtkOutput

            For optional arguments, see the `Cell.plot` method
        """
        # if fig is None:
        #     import matplotlib.pyplot as plt
        #     self.plot( plt )
        #     plt.show()
        #     return

        if isinstance( fig, str ):
            if fig.endswith( ".vtk" ):
                return self.plot_vtk( fig, **kwargs )
            raise ValueError( "Unhandled file type" )

        return self.plot_pyplot( fig, **kwargs )

    def _update_internal_attributes( self ):
        """ 
            compute self._binding_module, self._acceleration_structure and self._base_cell if not already done
        """

        # binding module
        dtype = self.dtype
        ndim = self.ndim
        if dtype is None or ndim is None:
            return False
        self._binding_module = module_for( 'sdot', use_arch = True, scalar_type = dtype, nb_dims = ndim )

        # acceleration structure
        if self._acceleration_structure is None or self._acceleration_structure.dtype != dtype or self._acceleration_structure.ndim != ndim:
            # set weights if not defined
            if self._weights is None:
                self._weights = PoomVec( np.zeros( self._positions.shape[ 0 ], dtype = numpy_dtype_for( dtype ) ) )

            # get dtype consistency. TODO: use PoomVecInst_Cast<...>
            if self._positions.dtype != dtype:
                self._positions = PoomVec( self._positions.as_ndarray.astype( numpy_dtype_for( dtype ) ) )
            if self._weights.dtype != dtype:
                self._weights = PoomVec( self._weights.as_ndarray.astype( numpy_dtype_for( dtype ) ) )

            #
            transformation_matrices = []
            for i, pt in enumerate( self._periodicity_transformations ):
                tm = pt.get( ndim )
                ti = np.linalg.inv( tm )
                transformation_matrices.append( tm )
                transformation_matrices.append( ti )
                for j in range( i ):
                    om = self._periodicity_transformations[ j ].get( ndim )
                    oi = np.linalg.inv( om )
                    transformation_matrices.append( om @ tm )
                    transformation_matrices.append( oi @ tm )
                    transformation_matrices.append( om @ ti )
                    transformation_matrices.append( oi @ ti )

            # make self._acceleration_structure
            self._acceleration_structure = self._binding_module.LowCountAccelerationStructure( self._positions._vec, self._weights._vec, transformation_matrices )

        # beginning of base cell
        self._base_cell = Cell( _cell = self._binding_module.Cell(), _binding_module = self._binding_module )
        nb_base_bnds = 0
        if self._boundaries is not None:
            ndim = self._binding_module.ndim()
            nb_base_bnds = len( self._boundaries )
            for n, bnd in enumerate( self._boundaries ):
                self._base_cell.cut_boundary( bnd[ : ndim ],  bnd[ ndim ], n )

        # density split between boundaries and value (possible modification of base cell)
        bnds, value = self._underlying_measure.boundary_split( ndim )
        # naughty
        from .distributions.ScaledImage import ScaledImage
        if not isinstance( value, ScaledImage ):
            self._density_under_final_boundaries = Expr( value )

        for n, bnd in enumerate( bnds ):
            self._base_cell.cut_boundary( bnd[ : ndim ],  bnd[ ndim ], nb_base_bnds + n )

        return True

