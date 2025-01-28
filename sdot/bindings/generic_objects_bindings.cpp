#include <tl/support/string/CompactReprReader.h>
#include <tl/support/string/to_string.h>
#include <sdot/support/binding_config.h>
#include <sdot/support/VtkOutput.h>

#include <sdot/symbolic/instructions/axis.h>
#include <sdot/symbolic/ExprData.h>
#include <sdot/symbolic/Expr.h>
 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
 
using namespace sdot;

std::tuple<Str,std::vector<ExprData>> ct_rt_split_of_list( const std::vector<Expr> &expr_list ) {
    Vec<ExprData> data_map;
    CompactReprWriter cw;
    cw.write_positive_int( expr_list.size() );

    for( const Expr &expr : expr_list )
        expr.inst->ct_rt_split( cw, data_map );

    std::vector<ExprData> rt_data;
    for( auto &p : data_map )
        rt_data.push_back( std::move( p ) );

    return { cw.str(), std::move( rt_data ) };
}

std::tuple<std::pair<Str,Str>,std::vector<Expr>> cell_splits_of_list( const std::vector<Expr> &expr_list, const std::vector<ExprData> &expr_data ) {
    TODO;
    return {};
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
        
        .def( "always_equal", []( const Expr &a, const Expr &b ) { return a.always_equal( b ); } )

        .def( "constant_value", []( const Expr &a ) { Opt<BigRational> v = a.constant_value(); bool valid( v ); FP64 value; if ( valid ) value = FP64( *v ); return std::make_tuple( valid, value ); } )
       
        .def( "subs", []( const Expr &expr, const std::vector<std::pair<Expr,Expr>> &symbol_map ) {
            std::map<RcPtr<Inst>,RcPtr<Inst>> map;
            for( const auto &p : symbol_map )
                map[ p.first.inst ] = p.second.inst;
            return expr.subs( map );
        } )

        .def( "apply", []( const Expr &a, const std::vector<std::pair<Str,Expr>> &b ) {
            try {
                return Expr( a.inst->apply( map_vec( b, []( const std::pair<Str,Expr> &e ) { return std::pair<Str,RcPtr<Inst>>{ e.first, e.second.inst }; } ) ) );
            } catch ( Inst::NaturalArgsError e ) {
                throw pybind11::value_error{ e.msg };
            }
        } )

        .def( "natural_args", []( const Expr &expr ) -> std::vector<std::vector<std::pair<Str,Expr>>> {
            Vec<Vec<std::pair<Str,RcPtr<Inst>>>> nargs;
            expr.inst->find_natural_args_rec( nargs );

            std::vector<std::vector<std::pair<Str,Expr>>> res;
            for( auto &narg : nargs ) {
                std::vector<std::pair<Str,Expr>> loc;
                for( auto &p : narg )
                    loc.push_back( std::pair<Str,Expr>{ p.first, p.second } );
                res.push_back( std::move( loc ) );
            }
            return res;
        } )

        .def( "set_natural_args", []( const Expr &expr, const std::vector<std::pair<Str,Expr>> &args ) {
            expr.inst->clear_natural_args();
            for( const auto &arg : args )
                expr.inst->add_natural_arg( arg.first, arg.second.inst );
        } )

        .def( "add", []( const Expr &a, const Expr &b ) { return a + b; } )
        .def( "sub", []( const Expr &a, const Expr &b ) { return a - b; } )
        .def( "mul", []( const Expr &a, const Expr &b ) { return a * b; } )
        .def( "div", []( const Expr &a, const Expr &b ) { return a / b; } )
        .def( "pow", []( const Expr &a, const Expr &b ) { return pow( a, b ); } )

        .def( "always_equal", []( const Expr &a, const Expr &b ) { return a.always_equal( b ); } )
        .def( "alternative", []( const Expr &cond, const std::vector<Expr> &list ) { return alternative( cond, list ); } )
        .def( "and_boolean", []( const Expr &a, const Expr &b ) { return and_boolean( a, b ); } )
        .def( "or_boolean", []( const Expr &a, const Expr &b ) { return or_boolean( a, b ); } )
        .def( "equal", []( const Expr &a, const Expr &b ) { return a == b; } )
        .def( "supeq", []( const Expr &a, const Expr &b ) { return a >= b; } )
        .def( "infeq", []( const Expr &a, const Expr &b ) { return a <= b; } )
        .def( "ceil", []( const Expr &a ) { return ceil( a ); } )
        .def( "frac", []( const Expr &a ) { return frac( a ); } )
        .def( "neq", []( const Expr &a, const Expr &b ) { return a != b; } )
        .def( "sup", []( const Expr &a, const Expr &b ) { return a > b; } )
        .def( "inf", []( const Expr &a, const Expr &b ) { return a < b; } )

        .def( "neg", []( const Expr &a ) { return - a; } )
        ;

    pybind11::class_<ExprData>( m, "ExprData" )
        .def( "__repr__", []( const ExprData &cell ) { return "ExprData"; } )
        ;

    m.def( "expr_list_from_compact_repr", expr_list_from_compact_repr );
    m.def( "ct_rt_split_of_list", ct_rt_split_of_list );
    m.def( "cell_splits_of_list", cell_splits_of_list );
    m.def( "axis", [&]( int n ) { return Expr( sdot::axis( n ) ); } );
}
