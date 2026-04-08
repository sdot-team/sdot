from .distributions.helpers.distribution_methods import _collect_attributes, ListOfTensorFields, TensorField, flat_tensor_list, unflat_tensor_list
from .driver import driver, encode_base62, tensor_conv_for
from .distributions.Distribution import Distribution
from .Bsp import Bsp

from pathlib import Path


class OtSolver:
    """
    """

    def __init__( self, bsp: Bsp, g: Distribution ):
        self.bsp = bsp
        self.g = g

        self.bindings = self._bindings()


    def write_vtk( self, filename: str ):
        flatten_g_content = flat_tensor_list( self.g )
        assert len( self.bsp.items ) == 1

        binding_inputs = []
        unflat_tensor_list( self.g, binding_inputs, list( flatten_g_content ) )

        self.bindings.write_vtk( filename, self.bsp.items[ 0 ], *binding_inputs )


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
                #include <sdot/nanobind_wrappers.h>
                #include <nanobind/stl/vector.h>
                #include <nanobind/stl/string.h>
                #include <sdot/geometry/Bsp.h>
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

                void write_vtk( std::string filename, BspType &bsp, FORWARD_ARGS ) {
                    FORWARD_TENSOR_CONV
                    auto primitive = PRIMITIVE_FUNC;
                    VtkOutput vo;
                    bsp.for_each_cell( primitive, [&]( auto &cell ) {
                        cell.display_vtk( vo );
                    } );
                    vo.save( filename );
                }

                NB_MODULE( SDOT_BINDING_NAME, m ) {
                    //m.def( "backward", &backward );
                    //m.def( "forward", &forward );
                    m.def( "write_vtk", &write_vtk );
                }
            """ )

        # get the binding
        geometry_dir = Path( __file__ ).parents[ 2 ] / "cpp" / "sdot" / "geometry"
        return driver.import_bindings( dylib_name, src_func, [
            geometry_dir / "VtkOutput.cpp"
        ] )
