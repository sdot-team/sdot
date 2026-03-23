#pragma once

#include "common_macros.h"
#include "common_types.h"
#include "ASSERT.h"
#include <algorithm>
#include <ostream>
#include <array>

namespace sdot {

/// view on strided data (strides in bytes, handles non-contiguous arrays)
template<class T,int ct_rank>
class TensorView {
public:
    using  Strides   = std::array<SI,ct_rank>;  ///< byte strides
    using  Extent    = std::array<PI,ct_rank>;
    using  RawPtr    = std::conditional_t<std::is_const_v<T>,const std::byte*,std::byte*>;

    /* */  TensorView( T *data, Extent ext, Strides str ) : extent( ext ), strides( str ), ptr( reinterpret_cast<RawPtr>( data ) ) {}
    /* */  TensorView( T *data, Extent ext ) : extent( ext ), strides( contiguous_strides( ext ) ), ptr( reinterpret_cast<RawPtr>( data ) ) {}
    /* */  TensorView( T *data, PI size ) : extent{ size }, strides{ sizeof( T ) }, ptr( reinterpret_cast<RawPtr>( data ) ) {}

    T&     operator()( PI i0, PI i1, PI i2 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] + i1 * strides[ 1 ] + i2 * strides[ 2 ] ); }
    T&     operator()( PI i0, PI i1 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] + i1 * strides[ 1 ] ); }
    T&     operator()( PI i0 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] ); }
    T&     operator()() const { return *reinterpret_cast<T *>( ptr ); }

    T&     operator[]( PI i0 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] ); }

    bool   empty     () const { return ct_rank == 0 ? false : std::none_of( extent.begin(), extent.end(), []( auto a ) { return a != 0; } ); }

    PI     size      ( PI d ) const { return extent[ d ]; }
    PI     size      () const { ASSERT( rank() == 1 ); return size( 0 ); }
    PI     rank      () const { return ct_rank; }

    auto   squeeze   ( PI axis ) const -> TensorView<T,ct_rank-1> {
        std::array<SI,ct_rank-1> nstrides;
        std::array<PI,ct_rank-1> nextent;
        for( PI i = 0; i < ct_rank-1; ++i ) {
            nstrides[ i ] = strides[ i + ( i >= axis ) ];
            nextent[ i ] = extent[ i + ( i >= axis ) ];
        }
        return { reinterpret_cast<T *>( ptr ), nextent, nstrides };
    }

    auto   row       ( PI index ) const -> TensorView<T,ct_rank-1> {
        std::array<SI,ct_rank-1> nstrides;
        std::array<PI,ct_rank-1> nextent;
        for( PI i = 1; i < ct_rank; ++i ) {
            nstrides[ i - 1 ] = strides[ i ];
            nextent[ i - 1 ] = extent[ i ];
        }
        return { reinterpret_cast<T *>( ptr + index * strides[ 0 ] ), nextent, nstrides };
    }

    static Strides contiguous_strides( const Extent &ext ) {
        Strides s;
        s[ ct_rank - 1 ] = sizeof( T );
        for( int i = ct_rank - 2; i >= 0; --i )
            s[ i ] = s[ i + 1 ] * ext[ i + 1 ];
        return s;
    }

private:
    Extent  extent;   ///<
    Strides strides;  ///< byte strides
    RawPtr  ptr;      ///<
};

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,d> &p ) {
    if constexpr( d == 0 )
        return os << p();
    else if constexpr ( d == 1 ) {
        for( sdot::PI i = 0; i < p.size(); ++i )
            os << ( i ? ", " : "" ) << p[ i ];
        return os;
    } else {
        for( sdot::PI i = 0; i < p.size( 0 ); ++i )
            os << "\n" << p.row( i );
        return os;
    }
}
