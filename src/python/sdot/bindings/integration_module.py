from .loader import get_build_dir, module_for
from ..Expr import Expr
import math

def integration_module( funcs, scalar_type, nb_dims ):
    # get summary
    ct_repr, rt_data = Expr.ct_rt_split_of_list( funcs )
    module_name = 'integration_' + ct_repr

    # generate .cpp and SConstruct files
    bd = get_build_dir( 'generated', ct_repr )
    cf = bd / ( module_name + ".cpp" )
    with open( cf, 'w' ) as f:
        f.write( '#include <sdot/support/binding_config.h>\n' )
        f.write( '#include <sdot/Cell.h>\n' )
        
        f.write( '#include <pybind11/pybind11.h>\n' )
        f.write( '#include <pybind11/numpy.h>\n' )
        f.write( '#include <pybind11/stl.h>\n' )

        f.write( 'using namespace sdot;\n' )

        f.write( 'using Array_TF = pybind11::array_t<SDOT_CONFIG_scalar_type, pybind11::array::c_style>;\n' )
        f.write( 'using Array_PI = pybind11::array_t<PI, pybind11::array::c_style>;\n' )
        f.write( 'static constexpr int nb_dims = SDOT_CONFIG_nb_dims;\n' )
        f.write( 'using Arch = sdot::SDOT_CONFIG_arch;\n' )
        f.write( 'using TF = SDOT_CONFIG_scalar_type;\n' )
        f.write( 'using Pt = Vec<TF, nb_dims>;\n' )

        f.write( 'struct PD_NAME( CutInfo ) {\n' )
        f.write( '    CutType type;\n' )
        f.write( '    Pt      p1;\n' )
        f.write( '    TF      w1;\n' )
        f.write( '    PI      i1;\n' )
        f.write( '};\n' )
        f.write( '\n' )
        f.write( 'struct PD_NAME( CellInfo ) {\n' )
        f.write( '    Pt p0;\n' )
        f.write( '    TF w0;\n' )
        f.write( '    PI i0;\n' )
        f.write( '};\n' )

        f.write( 'using TCell = Cell<Arch,TF,nb_dims,PD_NAME( CutInfo ),PD_NAME( CellInfo )>;\n' )
        f.write( 'using TCut = Cut<TF,nb_dims,PD_NAME( CutInfo )>;\n' )

        f.write( 'struct PD_NAME( Integration ) {\n' )
        f.write( '    void operator()( const Vec<Pt,nb_dims+1> &simplex ) {\n' )
        f.write( '        using namespace std;\n' )
        f.write( '        Eigen::Matrix<TF,nb_dims,nb_dims> M;\n' )
        f.write( '        for( PI r = 0; r < nb_dims; ++r )\n' )
        f.write( '            for( PI c = 0; c < nb_dims; ++c )\n' )
        f.write( '                M( r, c ) = simplex[ r + 1 ][ c ] - simplex[ 0 ][ c ];\n' )
        f.write( f'        out[ 0 ] += M.determinant() / { math.factorial( nb_dims ) };\n' )
        f.write( '    }\n' )
        f.write( f'    Vec<TF,{ len( funcs ) }> out;\n' )
        f.write( '};\n' )

        f.write( 'PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) {\n' )
        f.write( '    m.def( "cell_integral", []( TCell &cell, const std::vector<int> &rt_data ) -> std::vector<TF> {\n' )
        f.write( '        PD_NAME( Integration ) res;\n' )
        f.write( '        for( auto &v : res.out )\n' )
        f.write( '            v = 0;\n' )
        f.write( '        cell.simplex_split( res );\n' )
        f.write( '        return { res.out.begin(), res.out.end() };\n' )
        f.write( '    } );\n' )
        f.write( '}\n' )

    sf = bd / ( module_name + ".SConstruct" )
    with open( sf, 'w' ) as f:
        f.write( 'from sdot.bindings.construct import construct\n' )
        f.write( 'construct( Environment, VariantDir, ARGLIST, "' + module_name + '", [ "scalar_type", "nb_dims" ], [\n' )
        f.write( '    "' + str( cf ) + '",\n' )
        f.write( '] )\n' )

    module = module_for( module_name, base_dir = bd, scalar_type = scalar_type, nb_dims = nb_dims )
    return module, rt_data
