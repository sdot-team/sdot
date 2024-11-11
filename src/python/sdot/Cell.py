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
    def vertex_coords( self ):
        """ get list of coordinates for each vertex (stored in a numpy array) in the current base """
        return self._cell.vertex_coords( True )

    @property
    def vertex_refs( self ):
        """ get list of cut indices for each vertex (stored in a numpy array) in the current base """
        return self._cell.vertex_refs( True )

    @property
    def true_dimensionality( self ):
        """ Size of the base.
        
            at the beginning, all the cell starts with true_dimensionality == 0. A first cut will lead to true_dimensionality == 1. 
            If not in the same direction, true_dimensionality will be incremented, ...
            If true_dimensionality < ndims, coordinates are expressed in self.base
        """
        return self._cell.true_dimensionality()

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
        """ nb vertices in current dimensionality """
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
        """ internal base, used for vertex_coords if true_dimensionnality < nb_dims.
            It true_dimensionality == nb_dims, it return an identity matrix
        """
        return self._cell.base()
    
    @property
    def ndim( self ):
        return self._cell.ndim()
    
    @property
    def cuts( self ):
        return self._cell.cuts()
    
    def cut( self, dir, val, index = 0 ):
        return self._cell.cut( np.ascontiguousarray( dir ), val, int( index ) )

    def display_vtk( self, vtk_output ):
        return self._cell.display_vtk( vtk_output )

    def for_each_face( self, on_closed = None, on_2_rays = None, on_1_ray = None, on_free = None ):
        """
            Call args:
                on_closed( cut_refs, vertex_indices ), 
                on_2_rays( cut_refs, ray_1_refs, vertex_indices, ray_2_refs ) with vertex of ray 1 at the beginning, vertex of ray 2 at the end, 
                on_1_ray ( cut_refs, ray_refs ), 
                on_free  ( cut_refs ), 
        """
        if on_closed is None:
            on_closed = lambda cut_refs, vertex_indices: 0
        if on_2_rays is None:
            on_2_rays = lambda cut_refs, ray_1_refs, vertex_indices, ray_2_refs: 0
        if on_1_ray is None:
            on_1_ray = lambda cut_refs, ray_refs: 0
        if on_free is None:
            on_free = lambda cut_refs: 0
        return self._cell.for_each_face( on_closed, on_2_rays, on_1_ray, on_free )

    def plot_in_pyplot( self, fig ):
        """  
        
        """
        if self.ndim == 2:
            coords = self.vertex_coords @ self.base.T
            faces = list( self.closed_faces )
            for face in faces:
                if len( face ) == 0:
                    continue
                x = [ coords[ n, 0 ] for n in face ] + [ coords[ face[ 0 ], 0 ] ]
                y = [ coords[ n, 1 ] for n in face ] + [ coords[ face[ 0 ], 1 ] ]
                fig.plot( x, y )
            fig.axis('equal')
            return

        raise RuntimeError( f"TODO: plot for dim == { self.ndim }" )
        
