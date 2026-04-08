from .distributions.helpers.distribution_methods import _collect_attributes, ListOfTensorFields, TensorField, flat_tensor_list, unflat_tensor_list
from .driver import driver, encode_base62, tensor_conv_for
from .distributions.Distribution import Distribution
from .Polynomial import Polynomial
from .Cell import Cell
from .Bsp import Bsp

from pathlib import Path
import numpy

class OtSolver:
    """
    """

    def __init__( self, bsp: Bsp, g: Distribution ):
        self.bsp = bsp
        self.g = g

        self.bindings = self._bindings()


    @property
    def nb_diracs( self ):
        return self.bsp.nb_points


    @property
    def dim( self ):
        return self.bsp.dim


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
        res = numpy.zeros( [ 1 + n + n * n + n * n * n ] )
        self.bindings.get_poly_3( res, self.bsp.items[ 0 ], sorted_potentials, search_dirs, *self._inputs() )
        return Polynomial( 3, n, res )


    def write_vtk( self, sorted_potentials, filename: str ):
        self.bindings.write_vtk( filename, self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )


    def for_each_cell( self, sorted_potentials, func ):
        def on_cell( cell ):
            func( Cell( instance = cell ) )
        self.bindings.for_each_cell( on_cell, self.bsp.items[ 0 ], sorted_potentials, *self._inputs() )


    def plot( self, sorted_potentials, plotter = None ):
        import pyvista

        own_plotter = plotter is None
        if own_plotter:
            plotter = pyvista.Plotter( theme = pyvista.plotting.themes.DarkTheme() )
            if self.dim == 2:
                plotter.view_xy()

        def on_cell( cell ):
            cell.plot( plotter = plotter )

        self.for_each_cell( sorted_potentials, on_cell )

        if own_plotter:
            plotter.reset_camera()
            plotter.show()


    def _inputs( self ):
        flatten_g_content = flat_tensor_list( self.g )
        assert len( self.bsp.items ) == 1

        binding_inputs = []
        unflat_tensor_list( self.g, binding_inputs, list( flatten_g_content ) )

        return binding_inputs


    def _bindings( self ):
        ct_dim = self.g.dim if self.g.dim <= 4 else -1
        p_func, g_func, includes = type( self.g ).BaseVersion.primitive_function( self.g )
        dylib_name = f"ot_plan_{ encode_base62( "||".join( [ p_func, g_func ] + includes ) ) }_{ ct_dim }d_{ driver.normalized_dtype }_{ driver.normalized_device_type }"

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
                "PRIMITIVE_GRAD": g_func,
                "SDOT_INCLUDES": str.join( "\n", [ f"#include <{ include }>" for include in includes ] ),
            }

            return driver.cpp_src( m, """
                #include <sdot/support/TaylorScalar.h>
                #include <sdot/support/vector_map.h>
                #include <sdot/nanobind_wrappers.h>
                #include <nanobind/stl/function.h>
                #include <nanobind/stl/vector.h>
                #include <nanobind/stl/string.h>
                #include <sdot/geometry/Bsp.h>
                #include <sdot/support/P.h>
                SDOT_INCLUDES

                namespace nb = nanobind;
                using namespace sdot;

                using NA = nanobind::device::SDOT_NANOBIND_ARCH;
                using TF = SDOT_SCALAR_TYPE;

                using Arch = ArchFor<NA>::type;
                using AF = nb::ndarray<const TF,NA>; // const array
                using MF = nb::ndarray<TF,NA>; // mutable array

                using BspType = Bsp<PI,TF,2,Arch>;

                //
                // void forward( FORWARD_ARGS ) {
                //     FORWARD_TENSOR_CONV
                //     ot_plan_1d_forward( f_positions, f_weights, PRIMITIVE_FUNC, distances, barycenters, potentials, cuts );
                // }

                void get_search_dir( MF _search_dir, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dir = tensor_view_1( _search_dir );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
                        TF der = 0;
                        TF measure = cell.for_each_cut_with_measure( [&]( const auto &cut, const TF cut_measure ) {
                            if ( cut.info.global_dirac_index == PI( -1 ) )
                                return;
                            der += cut_measure / ( 2 * norm_2( cut.info.dirac_position - cell.info.dirac_position ) );
                        } );
                        search_dir[ cell.info.local_dirac_index ] = ( measure - cell.info.dirac_weight ) / der;
                    } );
                }

                void get_system( MF _M, MF _V, BspType &bsp, AF _sorted_potentials, const std::vector<AF> &_search_dirs, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dirs = vector_map( _search_dirs, []( AF af ) { return tensor_view_1( af ); } );
                    auto M = tensor_view_2( _M );
                    auto V = tensor_view_1( _V );
                    const PI n = V.size();
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    std::vector<TF> loc_ders( n );
                    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
                        for( TF &d : loc_ders )
                            d = 0;
                        TF measure = cell.for_each_cut_with_measure( [&]( const auto &cut, const TF cut_measure ) {
                            if ( cut.info.global_dirac_index == PI( -1 ) )
                                return;
                            const TF coeff = cut_measure / ( 2 * norm_2( cut.info.dirac_position - cell.info.dirac_position ) );
                            for( PI r = 0; r < n; ++r )
                                loc_ders[ r ] += ( search_dirs[ r ][ cell.info.local_dirac_index ] - search_dirs[ r ][ cut.info.local_dirac_index ] ) * coeff;
                        } );

                        for( PI r = 0; r < n; ++r ) {
                            for( PI c = 0; c < n; ++c )
                                M( r, c ) += loc_ders[ c ] * loc_ders[ r ];
                            V[ r ] += ( measure - cell.info.dirac_weight ) * loc_ders[ r ];
                        }
                    } );
                }

                void get_poly_3( MF _poly, BspType &bsp, AF _sorted_potentials, const std::vector<AF> &_search_dirs, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    auto search_dirs = vector_map( _search_dirs, []( AF af ) { return tensor_view_1( af ); } );
                    auto poly = tensor_view_1( _poly );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    const PI order = 3;

                    using Scalar = TaylorScalar<TF>; // ??
                    std::vector<Scalar> coefficients( search_dirs.size() );
                    for( PI i = 0; i < search_dirs.size(); ++i )
                        coefficients[ i ] = Scalar::variable( search_dirs.size(), i );

                    auto modified_potentials = [&]( PI index ) {
                        Scalar res = sorted_potentials[ index ];
                        for( PI i = 0; i < search_dirs.size(); ++i )
                            res += search_dirs[ i ][ index ] * coefficients[ i ];
                        return res;
                    };

                    Scalar res = 0;
                    bsp.for_each_cell( primitive, modified_potentials, [&]( auto &cell ) {
                        auto err = cell.measure() - cell.info.dirac_weight;
                        res += err * err;
                    } );

                    PI i = 0;
                    poly[ i++ ] = res.c0;
                    for( TF v : res.c1 )
                        poly[ i++ ] = v;
                    for( TF v : res.c2 )
                        poly[ i++ ] = v;
                    for( TF v : res.c3 )
                        poly[ i++ ] = v;
                }

                void for_each_cell( const std::function<void( BspType::Cell &cell )> &function, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
                        function( cell );
                    } );
                }

                void write_vtk( std::string filename, BspType &bsp, AF _sorted_potentials, FORWARD_ARGS ) {
                    auto sorted_potentials = tensor_view_1( _sorted_potentials );
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    VtkOutput vo;
                    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
                        cell.display_vtk( vo );
                    } );
                    vo.save( filename );
                }

                NB_MODULE( SDOT_BINDING_NAME, m ) {
                    m.def( "get_search_dir", &get_search_dir );
                    m.def( "for_each_cell", &for_each_cell );
                    m.def( "get_system", &get_system );
                    m.def( "get_poly_3", &get_poly_3 );
                    m.def( "write_vtk", &write_vtk );
                }
            """ )

        # get the binding
        geometry_dir = Path( __file__ ).parents[ 2 ] / "cpp" / "sdot" / "geometry"
        return driver.import_bindings( dylib_name, src_func, [
            geometry_dir / "VtkOutput.cpp"
        ] )
