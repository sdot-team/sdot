#pragma once

#include "TensorView.h"
#include <stdexcept>

namespace sdot {

template<int rank,class Arch>
class DynamicAxis {
public:
    using        Sizes      = TensorView<PI64,rank,Arch>;

    /**/         DynamicAxis( Sizes sizes, PI capacity, Sizes src ) : DynamicAxis( sizes, capacity ) { sizes.get_data_from( src ); } // convenience ctor
    /**/         DynamicAxis( Sizes sizes, PI capacity ) : capacity( capacity ), sizes( sizes ) {}

    // slicing/subparts
    auto         operator() ( auto...indices ) const { constexpr int new_rank = rank - int( sizeof...( indices ) ); return DynamicAxis<new_rank,Arch>( sizes.partial( indices... ), capacity ); }
    auto         row        ( auto index ) const { return operator()( index ); }

    // assuming rank == 0
    PI           operator++ () { if ( ++sizes() > capacity ) overflow(); return sizes(); }
    PI           operator++ ( int ) { PI res = sizes(); ++( *this ); return res; }
    DynamicAxis& operator=  ( PI value ) { if ( value > capacity ) overflow(); sizes() = value; return *this; }
    operator     PI         () const { return sizes(); }

// private:
    void         overflow   () { throw std::runtime_error( "DynamicAxis: capacity exceeded" ); }

    const PI     capacity;
    Sizes        sizes;
};

} // namespace sdot
