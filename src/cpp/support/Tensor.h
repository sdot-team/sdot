#pragma once

#include "TensorView.h"
#include <vector>

namespace sdot {

/// owning tensor — contiguous row-major by construction, strides stored (no recomputation on access)
template<class T,int ct_rank>
class Tensor {
public:
    using       CTV       = TensorView<const T,ct_rank>;
    using       TV        = TensorView<T,ct_rank>;
    using       Strides   = typename TV::Strides;
    using       Extent    = typename TV::Extent;
    using       Storage   = std::vector<T>;
    using       Byte      = std::byte;

    T_U         Tensor    ( std::initializer_list<std::initializer_list<U>> m ) : Tensor( sdot::Extent(), { m.size(), m.begin()->size() } ) { PI e = 0; for( const auto &l : m ) { PI c = 0; for( const auto &v : l ) operator()( e, c++ ) = v; ++e; } }
    T_U         Tensor    ( std::initializer_list<U> l ) : Tensor( sdot::Extent(), { l.size() } ) { PI cpt = 0; for( const auto &v : l ) operator()( cpt++ ) = v; }
    /* */       Tensor    ( sdot::Extent, Extent ext ) : extent( ext ), strides( TV::contiguous_strides( ext ) ), storage( total_size( ext ) ) {}
    /* */       Tensor    () { for( PI i = 0; i < ct_rank; ++i ) { strides[ i ] = 0; extent[ i ] = 0; } }

    CTV         view      () const { return { storage.data(), extent, strides }; }
    TV          view      () { return { storage.data(), extent, strides }; }

    operator    CTV       () const { return view(); }
    operator    TV        () { return view(); }

    const T&    operator()( PI i0, PI i1, PI i2 ) const { return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] + i2 * strides[ 2 ] ); }
    const T&    operator()( PI i0, PI i1 ) const { return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] ); }
    const T&    operator()( PI i0 ) const { return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] ); }
    const T&    operator()() const { return *reinterpret_cast<const T *>( bytes() ); }

    T&          operator()( PI i0, PI i1, PI i2 ) { return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] + i2 * strides[ 2 ] ); }
    T&          operator()( PI i0, PI i1 ) { return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] ); }
    T&          operator()( PI i0 ) { return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] ); }
    T&          operator()() { return *reinterpret_cast<T *>( bytes() ); }

    const T&    operator[]( PI i0 ) const { return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] ); }
    T&          operator[]( PI i0 )       { return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] ); }

    auto        row       ( PI index ) const { return CTV( *this ).row( index ); }
    auto        row       ( PI index ) { return TV ( *this ).row( index ); }

    auto        squeeze   ( PI axis  ) const { return CTV( *this ).squeeze( axis ); }
    auto        squeeze   ( PI axis  ) { return TV ( *this ).squeeze( axis ); }

    bool        empty     () const { return std::none_of( extent.begin(), extent.end(), []( auto a ) { return a != 0; } ); }
    PI          size      ( PI d ) const { return extent[ d ]; }
    PI          size      () const { static_assert( ct_rank == 1 ); return size( 0 ); }

    const T*    data      () const { return storage.data(); }
    T*          data      () { return storage.data(); }

private:
    static PI   total_size( const Extent &ext ) { PI s = 1; for( auto e : ext ) s *= e; return s; }

    const auto* bytes     () const { return reinterpret_cast<const std::byte *>( storage.data() ); }
    auto*       bytes     (){ return reinterpret_cast<std::byte *>( storage.data() ); }

    Extent      extent;
    Strides     strides;
    Storage     storage;
};

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::Tensor<T,d> &t ) {
    return os << sdot::TensorView<const T,d>( t );
}
