from .bindings.integration_module import integration_module
from .bindings.loader import module_for
from .Expr import Expr

from types import ModuleType
import numpy as np

class Cell:
    """ wrapper around cpp sdot::Cell class """

    def __init__( self, ndim = None, dtype = None, _cell = None, _binding_module = None ):
        """ _cell is the sdot::Cell class instance. If not specified, __init__ makes a new one using `ndim` and `dtype`.
            By default, dtype = np.double, ndim = 2
        """

        if _cell:
            assert _binding_module is not None
            self._binding_module = _binding_module
            self._cell = _cell
            return

        if ndim is None:
            ndim = 2

        if dtype is None:
            dtype = "FP64"

        self._binding_module = module_for( 'sdot', use_arch = True, scalar_type = dtype, nb_dims = ndim )
        self._cell = self._binding_module.Cell()

    def __repr__( self ):
        return self._cell.__repr__()

    def integral( self, funcs = '1', underlying_measure_override = None, underlying_radial_function_override = None ):
        """ integral of a symbolic function or a list of symbolic functions """
        if not isinstance( funcs, list ):
            return self.integral( [ funcs ], underlying_measure_override, underlying_radial_function_override )[ 0 ]
        
        # module généré
        module, rt_data = integration_module( funcs, self._binding_module.dtype(), self._binding_module.ndim() )
        return module.cell_integral( self._cell, rt_data )

    @property
    def true_dimensionality( self ):
        """ Size of the base.
        
            at the beginning, all the cell starts with true_dimensionality == 0. A first cut will lead to true_dimensionality == 1. 
            If not in the same direction, true_dimensionality will be incremented, ...
            If true_dimensionality < ndims, coordinates are expressed in self.base
        """
        return self._cell.true_dimensionality()

    @property
    def vertex_coords_td( self ):
        """ get list of coordinates for each vertex (stored in a numpy array) in the current base """
        return self._cell.vertex_coords( True )

    @property
    def vertex_coords( self ):
        """ get list of coordinates for each vertex (stored in a numpy array) """
        return self._cell.vertex_coords( False )

    @property
    def vertex_refs_td( self ):
        """ get list of cut indices for each vertex (stored in a numpy array) in the current base """
        return self._cell.vertex_refs( True )

    @property
    def vertex_refs( self ):
        """ get list of cut indices for each vertex (stored in a numpy array) in the current base """
        return self._cell.vertex_refs( False )

    @property
    def nb_active_cuts( self ):
        """ beware: if remove_inactive_cuts() has not been called, this procedure will have first to compute the active cuts (may take some time) """
        return self._cell.nb_active_cuts()

    @property
    def nb_stored_cuts( self ):
        """ may be different from nb_active_cuts if remove_inactive_cuts() has not been called """
        return self._cell.nb_stored_cuts()

    @property
    def nb_vertices_td( self ):
        """ nb vertices in current "true" dimensionality """
        return self._cell.nb_vertices( True )

    @property
    def nb_vertices( self ):
        return self._cell.nb_vertices( False )

    @property
    def bounded( self ):
        return self._cell.bounded()

    @property
    def empty( self ):
        return self._cell.empty()
    
    # @property
    # def closed_faces( self ):
    #     return self._cell.closed_faces()
    
    @property
    def base( self ):
        """ internal base, used to get coordinates and directions if true_dimensionnality < nb_dims.
            It true_dimensionality == nb_dims, it return an identity matrix
        """
        return self._cell.base()
    
    @property
    def ndim( self ):
        return self._cell.ndim()
    
    @property
    def cuts( self ):
        return self._cell.cuts()
    
    def cut_boundary( self, dir, val, index = 0 ):
        return self._cell.cut_boundary( np.ascontiguousarray( dir ), val, int( index ) )

    def cut( self, dir, val, index = 0 ):
        return self._cell.cut( np.ascontiguousarray( dir ), val, int( index ) )

    def display_vtk( self, vtk_output ):
        return self._cell.display_vtk( vtk_output )

    # def for_each_face( self, on_face ):
    #     """
    #         Call args:
    #             on_face( cut_refs, vertex_indices, ray_refs_list ),
    #     """
    #     return self._cell.for_each_face( on_face )

    def for_each_edge( self, on_edge ):
        """
            on_edge( edge_refs, vertex_indices )
                * edge_refs = [ num_cut_0, ... ]
                * vertex_indices => 
                   - size = 2 for a regular edge
                   - size = 1 for a ray

        """
        return self._cell.for_each_edge( on_edge, False )

    def for_each_edge_td( self, on_edge ):
        """
            on_edge( edge_refs, vertex_indices )
                * edge_refs = [ num_cut_0, ... ]
                * vertex_indices => 
                   - size = 2 for a regular edge
                   - size = 1 for a ray

        """
        return self._cell.for_each_edge( on_edge, True )

    def ray_dir( self, ray_refs, base_vertex ):
        return self._cell.ray_dir( ray_refs, base_vertex )

    def plot_pyplot( self, fig, color = 'black', linestyle = '-', linewidth = 2, low_dim_linewidth_coeff = 0.25, ray_length = 0.2, ray_color = None, ray_linestyle = '--', free_color = None, free_linestyle = ':' ):
        """
            ray_length is the length used to display infinite edges.
        """

        # check fig properties
        if isinstance( fig, ModuleType ):
            fig.axis( 'equal' )

        #
        coords = self.vertex_coords_td @ self.base

        # find a complementary base
        cbase = list( np.eye( self.ndim ) )
        for r in range( self.ndim ):
            for c in self.base:
                cbase[ r ] -= np.dot( cbase[ r ], c ) * c
        cbase.sort( key = np.linalg.norm )
        cbase = cbase[ self.true_dimensionality : ]

        # helper to plot
        def pl( array, gtype ):
            array = np.array( array )
            xyz = [ array[ :, d ] for d in range( min( self.ndim, 3 ) ) ]
            kwa = { 'linewidth': linewidth }
            if self.true_dimensionality < self.ndim and gtype != 'free':
                kwa[ 'linewidth' ] *= low_dim_linewidth_coeff

            if gtype == 'free':
                return fig.plot( *xyz, **kwa, linestyle = free_linestyle, color = free_color or color )
            if gtype == 'ray':
                return fig.plot( *xyz, **kwa, linestyle = ray_linestyle, color = ray_color or color )
            if gtype == 'edge':
                return fig.plot( *xyz, **kwa, linestyle = linestyle, color = color )

        # for each vertex, plot free items, get the edges and the rays
        for c in coords:
            for cvec in cbase:
                pl( [ c - ray_length * cvec, c + ray_length * cvec ], 'free' )

        # plot edge
        def edge_func( refs, vertices ):
            if len( refs ) == 0:
                return

            # ray
            if len( vertices ) == 1:
                rd = self.ray_dir( refs, vertices[ 0 ] ) @ self.base
                rd /= np.linalg.norm( rd )
                bc = coords[ vertices[ 0 ] ]
                pl( [ bc, bc + ray_length * rd ], 'ray' )
                return
            
            # edge
            if len( vertices ) == 2:
                pl( [ coords[ vertices[ 0 ] ], coords[ vertices[ 1 ] ] ], 'edge' )
                return

            raise RuntimeError( "not an expected size for vertices" )

        self.for_each_edge_td( edge_func )

    def plot( self, fig = None, **kwargs ):
        """  
        """
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt
            if self.ndim == 3:
                fig = fig.figure()
                fig = fig.add_subplot( projection = '3d' )
            self.plot( fig )
            plt.show()
            return

        self.plot_pyplot( fig, **kwargs )

        # if self.true_dimensionality == 2:

        #     def disp_ray( ray_refs, vertex_index, rev ):
        #         print( "ray" )
        #         dir = self.ray_dir( ray_refs, vertex_index ) @ self.base.T
        #         b_c = coords[ vertex_index ]
        #         d_c = b_c + ray_length * dir / np.linalg.norm( dir )
        #         x = [ b_c[ 0 ], d_c[ 0 ] ]
        #         y = [ b_c[ 1 ], d_c[ 1 ] ]
        #         if rev:
        #             x.reverse()
        #             y.reverse()
        #         fig.plot( x, y, ray_linestyle or linestyle, color = ray_color or color )

        #     def on_face( edge_refs, vertex_indices, ray_refs ):
        #         print( "cut_refs" )

        #         if len( ray_refs ) >= 1:
        #             disp_ray( ray_refs[ 0 ], vertex_indices[ 0 ], 1 )

        #         if len( vertex_indices ) >= 2:
        #             x = [ coords[ n, 0 ] for n in vertex_indices ]
        #             y = [ coords[ n, 1 ] for n in vertex_indices ]
        #             if len( ray_refs ) == 0: # closed
        #                 x.append( x[ 0 ] )
        #                 y.append( y[ 0 ] )
        #             fig.plot( x, y, linestyle, color = color )

        #         if len( ray_refs ) >= 2:
        #             disp_ray( ray_refs[ 1 ], vertex_indices[ -1 ], 0 )

        #     self.for_each_face( on_face )
        #     return

        # raise RuntimeError( f"TODO: plot for dim == { self.ndim }" )
        
