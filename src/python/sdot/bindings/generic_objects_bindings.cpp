#include <tl/support/string/CompactReprReader.h>
#include <tl/support/string/to_string.h>
#include <sdot/support/binding_config.h>
#include <sdot/support/VtkOutput.h>
#include <sdot/symbolic/ExprData.h>
#include <sdot/symbolic/Expr.h>
 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace sdot;

std::tuple<Str,std::vector<ExprData>> ct_rt_split_of_list( const std::vector<Expr> &expr_list ) {
    Vec<std::pair<const Inst *,ExprData>> data_map;
    CompactReprWriter cw;
    cw.write_positive_int( expr_list.size() );
    for( const Expr &expr : expr_list )
        expr.inst->ct_rt_split( cw, data_map );

    std::vector<ExprData> rt_data;
    for( auto &p : data_map )
        rt_data.push_back( std::move( p.second ) );

    return { cw.str(), std::move( rt_data ) };
}

std::vector<Expr> expr_list_from_compact_repr( const Str &str ) {
    CompactReprReader cr( str );

    PI size( cr.read_positive_int() );

    std::vector<Expr> res;
    for( PI i = 0; i < size; ++i )
        res.push_back( Expr{ Inst::read_from( cr ) } );
    return res;
}

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    pybind11::class_<VtkOutput>( m, "VtkOutput" )
        .def( pybind11::init<>() )
        .def( "save", []( VtkOutput &self, Str filename ) { self.save( filename ); } )
        ;

    pybind11::class_<Expr>( m, "Expr" )
        .def( pybind11::init( []( FP64 v ) { return Expr( BigRationalFrom<FP64>::create( v ) ); } ) )
        .def( pybind11::init<Str>() )
        .def( pybind11::init<SI>() )
        .def( "__repr__", []( const Expr &cell ) { return to_string( cell ); } )
       
        .def( "add", []( const Expr &a, const Expr &b ) { return a + b; } )
        .def( "sub", []( const Expr &a, const Expr &b ) { return a - b; } )
        .def( "mul", []( const Expr &a, const Expr &b ) { return a * b; } )
        .def( "div", []( const Expr &a, const Expr &b ) { return a / b; } )
        .def( "pow", []( const Expr &a, const Expr &b ) { return pow( a, b ); } )
        ;

    m.def( "ct_rt_split_of_list", ct_rt_split_of_list );
    m.def( "expr_list_from_compact_repr", expr_list_from_compact_repr );
    // m.def( "Expr_from_image", []( const Array_TF &img ) {

    // } );

    //     return module.( array, tr_coords, interpolation_order )
}
