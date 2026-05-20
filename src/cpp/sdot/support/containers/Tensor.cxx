#pragma once

#include "Tensor.h"

#include <algorithm>

namespace sdot {

#define UTP template<class T,int ct_rank,class Arch>
#define DTP Tensor<T,ct_rank,Arch>

UTP template<class U>
DTP::Tensor( std::initializer_list<std::initializer_list<std::initializer_list<U>>> m ) : Tensor( sdot::Shape(), { m.size(), m.begin()->size(), m.begin()->begin()->size() } ) {
    PI c0 = 0;
    for( const auto &v0 : m ) {
        PI c1 = 0;
        for( const auto &v1 : v0 ) {
            PI c2 = 0;
            for( const auto &v2 : v1 ) {
                operator()( c0, c1, c2 ) = v2;
                ++c2;
            }
            ++c1;
        }
        ++c0;
    }
}

UTP template<class U>
DTP::Tensor( std::initializer_list<std::initializer_list<U>> m ) : Tensor( sdot::Shape(), { m.size(), m.begin()->size() } ) {
    PI c0 = 0;
    for( const auto &v0 : m ) {
        PI c1 = 0;
        for( const auto &v1 : v0 )
            operator()( c0, c1++ ) = v1;
        ++c0;
    }
}

UTP template<class U>
DTP::Tensor( std::initializer_list<U> l ) : Tensor( sdot::Shape(), { l.size() } ) {
    PI cpt = 0;
    for( const auto &v : l )
        operator()( cpt++ ) = v;
}

UTP template<class U>
DTP::Tensor( std::span<U> l ) : Tensor( sdot::Shape(), { l.size() } ) {
    PI cpt = 0;
    for( const auto &v : l )
        operator()( cpt++ ) = v;
}

UTP DTP::Tensor( sdot::Shape, Extent ext ) : extent( ext ), strides( TV::contiguous_strides( ext ) ), storage( total_size( ext ) ) {
}

UTP DTP::Tensor( Rank, PI rank ) : extent( Size(), rank, 0 ), strides( Size(), rank, 0 ), storage( 0 ) {
}

UTP template<class T2,int ct_rank2>
DTP::Tensor( const TensorView<T2,ct_rank2,Arch> &src ) : Tensor( sdot::Shape(), src.sizes() ) {
    if ( src.is_contiguous() ) {
        std::copy( src.data(), src.data() + size(), data() );
    } else {
        src.for_each_index( [&]( const auto &idx ) {
            view()( idx ) = src( idx );
        } );
    }
}

UTP DTP::CTV DTP::view() const {
    return { Arch::raw_ptr( storage ), extent, strides };
}

UTP DTP::TV DTP::view() {
    return { Arch::raw_ptr( storage ), extent, strides };
}

UTP DTP::operator CTV() const {
    return view();
}

UTP DTP::operator TV() {
    return view();
}

UTP const T& DTP::operator()( PI i0, PI i1, PI i2 ) const {
    return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] + i2 * strides[ 2 ] );
}

UTP const T& DTP::operator()( PI i0, PI i1 ) const {
    return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] );
}

UTP const T& DTP::operator()( PI i0 ) const {
    return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] );
}

UTP const T& DTP::operator()() const {
    return *reinterpret_cast<const T *>( bytes() );
}

UTP T& DTP::operator()( PI i0, PI i1, PI i2 ) {
    return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] + i2 * strides[ 2 ] );
}

UTP T& DTP::operator()( PI i0, PI i1 ) {
    return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] + i1 * strides[ 1 ] );
}

UTP T& DTP::operator()( PI i0 ) {
    return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] );
}

UTP T& DTP::operator()() {
    return *reinterpret_cast<T *>( bytes() );
}

UTP const T& DTP::operator()( const auto &indices ) const { return CTV( *this )( indices ); }
UTP T&       DTP::operator()( const auto &indices )       { return TV(  *this )( indices ); }

UTP void DTP::for_each_index( auto &&func, PI sub ) const { CTV( *this ).for_each_index( FORWARD( func ), sub ); }
UTP auto DTP::unsqueeze()     const { return CTV( *this ).unsqueeze(); }
UTP bool DTP::is_contiguous() const { return CTV( *this ).is_contiguous(); }

UTP const T& DTP::operator[]( PI i0 ) const {
    return *reinterpret_cast<const T *>( bytes() + i0 * strides[ 0 ] );
}

UTP T& DTP::operator[]( PI i0 ) {
    return *reinterpret_cast<T *>( bytes() + i0 * strides[ 0 ] );
}

UTP auto DTP::row( PI index ) const {
    return CTV( *this ).row( index );
}

UTP auto DTP::row( PI index ) {
    return TV( *this ).row( index );
}

UTP auto DTP::squeeze( PI axis ) const {
    return CTV( *this ).squeeze( axis );
}

UTP auto DTP::squeeze( PI axis ) {
    return TV( *this ).squeeze( axis );
}

UTP typename DTP::Extent DTP::sizes() const {
    return extent;
}

UTP bool DTP::empty() const {
    if ( rank() == 0 )
        return false;
    for ( PI i = 0; i < sizes().size(); ++i )
        if ( extent[ i ] == 0 )
            return true;
    return false;
}

UTP PI DTP::size( PI d ) const {
    return extent[ d ];
}

UTP PI DTP::size() const {
    return total_size( extent );
}

UTP const T* DTP::data() const {
    return Arch::raw_ptr( storage );
}

UTP T* DTP::data() {
    return Arch::raw_ptr( storage );
}

UTP const T* DTP::begin() const {
    return data();
}

UTP T* DTP::begin() {
    return data();
}

UTP const T* DTP::end() const {
    static_assert( ct_rank == 1 );
    return data() + size( 0 );
}

UTP T* DTP::end() {
    static_assert( ct_rank == 1 );
    return data() + size( 0 );
}

UTP PI DTP::total_size( const Extent &ext ) {
    PI s = 1;
    for( auto e : ext )
        s *= e;
    return s;
}

UTP const auto* DTP::bytes() const {
    return reinterpret_cast<const std::byte *>( data() );
}

UTP auto* DTP::bytes() {
    return reinterpret_cast<std::byte *>( data() );
}

#undef UTP
#undef DTP

} // namespace sdot

template<class T,int ct_rank,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::Tensor<T,ct_rank,Arch> &t ) {
    return os << sdot::TensorView<const T,ct_rank,Arch>( t );
}
