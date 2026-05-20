#include "../../src/cpp/sdot/support/hardware/MemorySpace_CpuRam.h"
#include "../../src/cpp/sdot/support/containers/TensorView.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;

struct T0 {
    int  operator()( auto... ) const { TODO; }
    int  operator[]( auto... ) const { TODO; }
    auto shape     () const { return std::vector<std::size_t>{}; }
    int  item      () const { return value; }
    int  value;
};

struct T1 {
    int  operator()( auto index ) const { return values[ index ]; }
    int  operator()( auto... ) const { TODO; }
    int  operator[]( auto index ) const { return values[ index ]; }
    auto shape     () const { return std::vector<std::size_t>{ values.size() }; }
    int  item      () const { TODO; }
    std::vector<int> values;
};

struct T2 {
    auto operator()( auto index ) const { return T1{ .values = values[ index ] }; }
    int  operator()( auto... ) const { TODO; }
    int  operator[]( auto... ) const { TODO; }
    auto shape     () const { return std::vector<std::size_t>{ values.size(), values[ 0 ].size() }; }
    int  item      () const { TODO; }
    std::vector<std::vector<int>> values;
};

TEST_CASE( "TensorView", "" ) {
    SECTION( "no ct axis" ) {
        double data[] = { 1, 2, 3, 4 };
        using Strides = AxisValues<int,2>;
        using Shape = AxisValues<int,2>;
        TensorView<double,Shape,Strides,MemorySpace_CpuRam> t( data, Shape( Values(), 2, 2 ), Strides( Values(), 2 * sizeof( double ), sizeof( double ) ) );
        T2 t2{ .values = { { 1, 2 }, { 3, 4 } } };
        CHECK_REPR( t, t2 );
        CHECK_REPR( t( 0 ), t2( 0 ) );
        CHECK_REPR( t( 1 ), t2( 1 ) );
    }
}
