#pragma once

// #include <stdexcept>
#include "TensorView.h"
#include "P.h"

namespace sdot {

//
struct DynamicSizeException {
    PI          needed_size;
    std::string name;
};


//
template<int rank,class Arch>
class DynamicAxis {
public:
    using        Sizes         = TensorView<PI64,rank,Arch>;

    /**/         DynamicAxis   ( const char *name, Sizes sizes, PI capacity, Sizes src ) : DynamicAxis( name, sizes, capacity ) { sizes.get_data_from( src ); } // convenience ctor
    /**/         DynamicAxis   ( const char *name, Sizes sizes, PI capacity ) : capacity( capacity ), sizes( sizes ), name( name ) {}

    // slicing/subparts
    auto         operator()    ( auto...indices ) const { constexpr int new_rank = rank - int( sizeof...( indices ) ); return DynamicAxis<new_rank,Arch>( name, sizes.partial( indices... ), capacity ); }
    auto         row           ( auto index ) const { return operator()( index ); }

    // assuming rank == 0
    PI           post_increment( PI value ) { PI res = sizes(); operator=( res + value ); return res; }
    PI           operator++    () { PI res = sizes() + 1; operator=( res ); return res; }
    PI           operator++    ( int ) { PI res = sizes(); operator=( res + 1 ); return res; }
    DynamicAxis& operator=     ( PI new_size ) { if ( new_size > capacity ) overflow( new_size ); sizes() = new_size; return *this; }
    operator     PI            () const { return sizes(); }

    void         overflow      ( PI needed_size ) { throw DynamicSizeException( needed_size, name ); }

    const PI     capacity;
    Sizes        sizes;
    const char*  name;
};

} // namespace sdot
