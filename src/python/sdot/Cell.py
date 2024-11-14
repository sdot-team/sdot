from .bindings.loader import sdot_module_for
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

        self._binding_module = sdot_module_for( scalar_type = dtype, nb_dims = ndim )
        self._cell = self._binding_module.Cell()

    def __repr__( self ):
        return self._cell.__repr__()

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
    
    @property
    def closed_faces( self ):
        return self._cell.closed_faces()
    
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

    def for_each_face( self, on_face ):
        """
            Call args:
                on_face( cut_refs, vertex_indices, ray_refs_list ),
        """
        return self._cell.for_each_face( on_face )

    def ray_dir( self, ray_refs, base_vertex ):
        return self._cell.ray_dir( ray_refs, base_vertex )

    def plot( self, fig, color = 'black', linestyle = '-', ray_length = 0.5, ray_color = None, ray_linestyle = '--' ):
        """  
            ray_size is used for infinite edges.
        """
        if self.true_dimensionality == 2:
            coords = self.vertex_coords_td @ self.base

            def disp_ray( ray_refs, vertex_index, rev ):
                dir = self.ray_dir( ray_refs, vertex_index ) @ self.base.T
                b_c = coords[ vertex_index ]
                d_c = b_c + ray_length * dir / np.linalg.norm( dir )
                x = [ b_c[ 0 ], d_c[ 0 ] ]
                y = [ b_c[ 1 ], d_c[ 1 ] ]
                if rev:
                    x.reverse()
                    y.reverse()
                fig.plot( x, y, ray_linestyle or linestyle, color = ray_color or color )

            def on_face( cut_refs, vertex_indices, ray_refs ):
                if len( ray_refs ) >= 1:
                    disp_ray( ray_refs[ 0 ], vertex_indices[ 0 ], 1 )

                if len( vertex_indices ) >= 2:
                    x = [ coords[ n, 0 ] for n in vertex_indices ]
                    y = [ coords[ n, 1 ] for n in vertex_indices ]
                    if len( ray_refs ) == 0: # closed
                        x.append( x[ 0 ] )
                        y.append( y[ 0 ] )
                    fig.plot( x, y, linestyle, color = color )

                if len( ray_refs ) >= 2:
                    disp_ray( ray_refs[ 1 ], vertex_indices[ -1 ], 0 )

            self.for_each_face( on_face )
            fig.axis('equal')
            return

        raise RuntimeError( f"TODO: plot for dim == { self.ndim }" )
        
