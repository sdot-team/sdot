from .distributions.helpers.distribution_methods import _collect_attributes, ListOfTensorFields, TensorField, flat_tensor_list, unflat_tensor_list
from .driver import driver, encode_base62, tensor_conv_for
from .distributions.Distribution import Distribution
from .LinearSolver import LinearSolver
from .Polynomial import Polynomial
from .Cell import Cell
from .Bsp import Bsp

from pathlib import Path
import numpy

class OtSolver:
    """
    """

    def __init__( self, bsp: Bsp, g: Distribution, linear_solver: LinearSolver = None ):
        self.sorted_potentials = None
        self.bsp = bsp
        self.g = g

        self.bindings = self._bindings()
        self._linear_solver = linear_solver or LinearSolver()

    @property
    def nb_diracs( self ):
        return self.bsp.nb_points

    @property
    def dim( self ):
        return self.bsp.dim

    def solve( self, max_iter = 100, error_tol = 1e-8, verbose = 0 ):
        if self.sorted_potentials is None:
            self.sorted_potentials = driver.zeros( [ self.nb_diracs ] )

        old_sorted_potentials = None
        for num_iter in range( max_iter ):
            for sub_iter in range( 10 ):
                try:
                    direction, error = self.linear_direction( self.sorted_potentials )
                    if verbose:
                        ic( error )
                except RuntimeError:
                    if old_sorted_potentials is None:
                        raise RuntimeError( "initial self.sorted_potentials are bad" )
                    self.sorted_potentials = ( old_sorted_potentials + self.sorted_potentials ) / 2
                    continue
                old_sorted_potentials = self.sorted_potentials.copy()
                self.sorted_potentials += direction
                break

            # ic( error )
            if error < error_tol:
                break


    def linear_direction( self, sorted_potentials ):
        m_rows, m_cols, m_vals, v_vals, error = self.bindings.system( self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )
        return self._linear_solver.solve( m_rows, m_cols, m_vals, v_vals, self.nb_diracs ), error


    def search_dirs( self, sorted_potentials ):
        res = [ numpy.zeros( [ self.nb_diracs ] ) for _ in range( 1 ) ]
        self.bindings.get_search_dirs( res, self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )
        return res


    def search_dir( self, sorted_potentials ):
        res = numpy.zeros( [ self.nb_diracs ] )
        self.bindings.get_search_dir( res, self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )
        return res


    def system( self, sorted_potentials, search_dirs ):
        n = len( search_dirs )
        M = numpy.zeros( [ n, n ] )
        V = numpy.zeros( [ n ] )
        self.bindings.get_system( M, V, self.bsp.items[ 0 ], sorted_potentials, search_dirs, *self._inputs() )
        return M, V


    def poly( self, sorted_potentials, search_dirs ) -> Polynomial:
        n = len( search_dirs )
        res = numpy.zeros( [ Polynomial.size_for( n, 3 ) ] )
        self.bindings.get_poly_3( res, self.bsp.items[ 0 ], sorted_potentials, search_dirs, *self._inputs() )
        return Polynomial( n, 3, res )


    def write_vtk( self, sorted_potentials, filename: str ):
        self.bindings.write_vtk( filename, self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )


    def for_each_cell( self, sorted_potentials, func ):
        def on_cell( cell ):
            func( Cell( instance = cell ) )
        self.bindings.for_each_cell( on_cell, self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )


    def plot( self, plotter = None ):
        import pyvista

        own_plotter = plotter is None
        if own_plotter:
            plotter = pyvista.Plotter( theme = pyvista.plotting.themes.DarkTheme() )
            if self.dim == 2:
                plotter.view_xy()

        def on_cell( cell ):
            cell.plot( plotter = plotter )

        self.for_each_cell( self.sorted_potentials, on_cell )

        if own_plotter:
            plotter.reset_camera()
            plotter.show()


    def _inputs( self ):
        flatten_g_content = flat_tensor_list( self.g )
        assert len( self.bsp.items ) == 1

        binding_inputs = []
        unflat_tensor_list( self.g, binding_inputs, list( flatten_g_content ) )

        return binding_inputs

    # def gc_eigen( ot_solver, pot, nb_to_keep=2, threshold=1e-10, use_eigs = False ):
    #     kept_dirs = []  # directions A-orthogonales de l'itération précédente
    #     for _ in range( 10 ):
    #         old_pot = pot.copy()
    #         for _ in range( 10 ):
    #             try:
    #                 z = [ ot_solver.search_dir( pot ) ]
    #                 # z = ot_solver.search_dirs( pot )
    #             except RuntimeError:
    #                 pot = ( pot + old_pot ) / 2
    #                 print( "bum" )
    #                 continue

    #             ic( numpy.linalg.norm( z[ 0 ] ) )

    #             #
    #             M_gs, V_gs = ot_solver.system( pot, z + kept_dirs )
    #             # poly = ot_solver.poly( pot, z + kept_dirs )

    #             X_gs = numpy.linalg.solve( M_gs, V_gs )
    #             # X_gs = poly.truncate( 2 ).argmin()
    #             for c, d in zip( X_gs, z + kept_dirs ):
    #                 pot += c * d

    #             # A-orthogonalise z contre les directions gardées (Gram-Schmidt en A-norme)
    #             n_kept = len( kept_dirs )
    #             coeffs = [ M_gs[ 0, i + 1 ] / M_gs[ i + 1, i + 1 ] for i in range( n_kept ) ]
    #             for c, d in zip( coeffs, kept_dirs ):
    #                 z[ 0 ] -= c * d

    #             #
    #             kept_dirs += z # .append( z )
    #             while len( kept_dirs ) > nb_to_keep:
    #                 kept_dirs.pop( 0 )

    #             break

    #     return pot

    # def gc( ot_solver, pot ):
    #     p = None   # direction conjuguée courante
    #     zg = None  # z_prev · g_prev, pour beta

    #     for _ in range( 10 ):
    #         z = ot_solver.search_dir( pot )   # z = P⁻¹ g
    #         print( numpy.linalg.norm( z ) )

    #         # z · g via system — sert de numérateur pour alpha et beta
    #         _, V = ot_solver.system( pot, [ z ] )
    #         zg_new = V[ 0 ]

    #         # direction conjuguée : p = z + beta * p_prev
    #         if p is not None:
    #             beta = zg_new / zg
    #             p = z + beta * p
    #         else:
    #             p = z.copy()

    #         # pas optimal : alpha = (z·g) / (p·A·p)
    #         M, _ = ot_solver.system( pot, [ p ] )
    #         alpha = zg_new / M[ 0, 0 ]
    #         pot = pot - alpha * p

    #         zg = zg_new

    #     return pot

    def _bindings( self ):
        ct_dim = self.g.dim if self.g.dim <= 4 else -1
        p_func, includes = type( self.g ).BaseVersion.primitive_function( self.g, False )
        dylib_name = f"ot_plan_{ encode_base62( "||".join( [ p_func ] + includes ) ) }_{ ct_dim }d_{ driver.normalized_dtype }_{ driver.normalized_device_type }"

        def src_func():
            # inputs of backward_args and forward_args
            # backward_args = []
            forward_args = []
            for d_name, d_data in [ ( "g", self.g ) ]:
                for a_name, a_data in _collect_attributes( type( d_data ) ):
                    if isinstance( a_data, ListOfTensorFields ):
                        # backward_args.append( ( "const std::vector<AF> &", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                        forward_args.append( ( "const std::vector<AF> &", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                    if isinstance( a_data, TensorField ):
                        # backward_args.append( ( "AF", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                        forward_args.append( ( "AF", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )

            # tensor_conv
            forward_tensor_conv = str.join( " ", [ tensor_conv_for( t, n, r ) for t, n, r in forward_args ] )

            m = {
                # "BACKWARD_TENSOR_CONV": backward_tensor_conv,
                "FORWARD_TENSOR_CONV": forward_tensor_conv,
                # "BACKWARD_ARGS": str.join( ", ", [ t + " _" + n for t, n, _ in backward_args ] ),
                "FORWARD_ARGS": str.join( ", ", [ t + " _" + n for t, n, _ in forward_args ] ),
                "PRIMITIVE_FUNC": p_func,
                # "PRIMITIVE_GRAD": g_func,
                "SDOT_INCLUDES": str.join( "\n", [ f"#include <{ include }>" for include in includes ] ),
            }

            return driver.cpp_src( m, """
                #include <sdot/support/vector_map.h>
                #include <sdot/nanobind_wrappers.h>
                #include <sdot/power_diagram.h>
                SDOT_INCLUDES

                #include <nanobind/stl/function.h>
                #include <nanobind/stl/vector.h>
                #include <nanobind/stl/string.h>
                #include <nanobind/stl/tuple.h>

                namespace nb = nanobind;
                using namespace sdot;

                using NA = nanobind::device::SDOT_NANOBIND_ARCH;
                using TF = SDOT_SCALAR_TYPE;

                using Arch = ArchFor<NA>::type;
                using AF = nb::ndarray<const TF,NA>; // const array
                using MF = nb::ndarray<TF,NA>; // mutable array

                using BspType = Bsp<PI,TF,2,Arch>;

                void _get_search_dirs( std::vector<MF> &_search_dirs, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dirs = vector_map( _search_dirs, []( MF af ) { return tensor_view_1( af ); } );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    get_search_dirs( search_dirs, bsp, sorted_potentials, primitive );
                }

                void _get_search_dir( MF _search_dir, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dir = tensor_view_1( _search_dir );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    get_search_dir( search_dir, bsp, sorted_potentials, primitive );
                }

                void _get_system( MF _M, MF _V, BspType &bsp, AF _sorted_potentials, const std::vector<AF> &_search_dirs, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dirs = vector_map( _search_dirs, []( AF af ) { return tensor_view_1( af ); } );
                    auto M = tensor_view_2( _M );
                    auto V = tensor_view_1( _V );
                    const PI n = V.size();
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    get_system( M, V, bsp, sorted_potentials, search_dirs, primitive );
                }

                void _get_poly_3( MF _poly, BspType &bsp, AF _sorted_potentials, const std::vector<AF> &_search_dirs, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dirs = vector_map( _search_dirs, []( AF af ) { return tensor_view_1( af ); } );
                    auto poly = tensor_view_1( _poly );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    get_poly_3( poly, bsp, sorted_potentials, search_dirs, primitive );
                }

                void _get_poly_4( MF _poly, BspType &bsp, AF _sorted_potentials, const std::vector<AF> &_search_dirs, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dirs = vector_map( _search_dirs, []( AF af ) { return tensor_view_1( af ); } );
                    auto poly = tensor_view_1( _poly );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    get_poly_4( poly, bsp, sorted_potentials, search_dirs, primitive );
                }

                void _for_each_cell( const std::function<void( BspType::Cell &cell )> &function, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    for_each_cell( bsp, function, sorted_potentials, primitive );
                }

                void _write_vtk( std::string filename, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    write_vtk( filename, bsp, sorted_potentials, primitive );
                }

                auto _system( BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;

                    return ot_system( bsp, sorted_potentials, primitive );
                }

                NB_MODULE( SDOT_BINDING_NAME, m ) {
                    m.def( "get_search_dirs", &_get_search_dirs );
                    m.def( "get_search_dir", &_get_search_dir );
                    m.def( "for_each_cell", &_for_each_cell );
                    m.def( "get_system", &_get_system );
                    m.def( "get_poly_3", &_get_poly_3 );
                    m.def( "get_poly_4", &_get_poly_4 );
                    m.def( "write_vtk", &_write_vtk );
                    m.def( "system", &_system );
                }
            """ )

        # get the binding
        geometry_dir = Path( __file__ ).parents[ 2 ] / "cpp" / "sdot" / "geometry"
        return driver.import_bindings( dylib_name, src_func, [
            geometry_dir / "VtkOutput.cpp"
        ] )
