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

UTP DTP::Tensor() {
    for( PI i = 0; i < ct_rank; ++i ) {
        strides[ i ] = 0;
        extent[ i ] = 0;
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

UTP bool DTP::empty() const {
    return std::none_of( extent.begin(), extent.end(), []( auto a ) { return a != 0; } );
}

UTP PI DTP::size( PI d ) const {
    return extent[ d ];
}

UTP PI DTP::size() const {
    static_assert( ct_rank == 1 );
    return size( 0 );
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
