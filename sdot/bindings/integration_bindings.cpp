// #include <tl/support/type_info/type_name.h>
#include <tl/support/string/to_string.h>
#include <sdot/support/VtkOutput.h>
#include <sdot/symbolic/ExprData.h>
#include <sdot/symbolic/Expr.h>
#include "config.h"
 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace sdot;

std::tuple<std::vector<Str>,std::vector<ExprData>> ct_rt_split_of_list( const std::vector<Expr> &expr_list ) {
    Vec<ExprData> data_map;
    std::vector<Str> ct_expr;
    for( const Expr &expr : expr_list ) {
        CompactReprWriter cw;
        expr.inst->ct_rt_split( cw, data_map );
        ct_expr.push_back( cw.str() );
    }

    std::vector<ExprData> rt_data;
    for( auto &p : data_map )
        rt_data.push_back( std::move( p.second ) );

    return { std::move( ct_expr ), std::move( rt_data ) };
}

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    pybind11::class_<VtkOutput>( m, "VtkOutput" )
        .def( pybind11::init<>() )
        .def( "save", []( VtkOutput &self, Str filename ) { self.save( filename ); } )
        ;

    pybind11::class_<Expr>( m, "Expr" )
        .def( pybind11::init<Str>() )
        .def( "__repr__", []( const Expr &cell ) { return to_string( cell ); } )
       
        .def( "add", []( const Expr &a, const Expr &b ) { return a + b; } )
        .def( "sub", []( const Expr &a, const Expr &b ) { return a - b; } )
        .def( "mul", []( const Expr &a, const Expr &b ) { return a * b; } )
        .def( "div", []( const Expr &a, const Expr &b ) { return a / b; } )
        .def( "pow", []( const Expr &a, const Expr &b ) { return pow( a, b ); } )
        ;

    m.def( "ct_rt_split_of_list", ct_rt_split_of_list );
}
