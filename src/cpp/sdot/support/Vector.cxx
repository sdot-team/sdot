#pragma once

#include <algorithm>
#include <utility>

#include "Vector.h"

namespace sdot {

#define UTPH template<class T,class Arch,int ct_size> HD
#define UTP template<class T,class Arch,int ct_size>
#define DTP Vector<T,Arch,ct_size>

UTPH DTP::Vector( const auto &values ) requires( requires { values.size(); } ) {
    ASSERT( values.size() == size() );
    auto iter = values.begin();
    for( PI i = 0; i < size(); ++i )
        new ( data() + i ) T( *( iter++ ) );
}

UTPH DTP::Vector( FillWith, auto &&...ctor_args ) {
    for( auto &v : *this )
        new ( &v ) T( ctor_args... );
}

UTPH DTP::Vector( Reserved ) {
}

UTPH DTP::Vector() {
    for( auto &v : *this )
        new ( &v ) T;
}

UTPH DTP::Vector( Values, auto &&...values ) : Vector( Reserved() ) {
    PI i = 0;
    auto append = [&]( auto &&value ) {
        new ( data() + i++ ) T( FORWARD( value ) );
    };
    ( append( FORWARD( values ) ), ... );
}

UTPH DTP::Vector( const Vector &that ) : Vector( Reserved() ) {
    for ( PI i = 0; i < that.size(); ++i )
        new ( data() + i ) T( that[ i ] );
}

UTPH DTP::Vector( Vector &&that ) noexcept {
    for ( PI i = 0; i < PI( ct_size ); ++i )
        new ( data() + i ) T( std::move( that[ i ] ) );
}

UTPH Vector<T,Arch,ct_size>& DTP::operator=( const Vector &that ) {
    if ( this != &that )
        for ( PI i = 0; i < size(); ++i )
            operator[]( i ) = that[ i ];
    return *this;
}

UTPH DTP& DTP::operator=( Vector &&that ) noexcept {
    if ( this != &that )
        for ( PI i = 0; i < PI( ct_size ); ++i )
            operator[]( i ) = std::move( that[ i ] );
    return *this;
}

UTPH DTP::~Vector() {
    for ( PI i = 0; i < size(); ++i )
        data()[ i ].~T();
}

UTPH const T& DTP::operator[]( PI index ) const {
    return data()[ index ];
}

UTPH T& DTP::operator[]( PI index ) {
    return data()[ index ];
}

// UTP bool DTP::operator<( const Vector &that ) const {
//     return std::ranges::lexicographical_compare( operator std::span<T>(), that.operator std::span<T>() );
// }

UTPH DTP DTP::with_func( auto &&func ) {
    Vector res;
    for ( PI i = 0; i < ct_size; ++i )
        new ( res.data() + i ) T( func( i ) );
    return res;
}

UTPH DTP DTP::zeros() {
    return with_func( [] HD ( PI ) { return T( 0 ); } );
}

UTPH DTP DTP::ones() {
    return with_func( [] HD ( PI ) { return T( 1 ); } );
}

UTPH DTP DTP::with_value_at( PI index, T value ) {
    return with_func( [=] HD ( PI i ) { return i == index ? value : T( 0 ); } );
}

UTPH auto DTP::with_pushed_value( T value ) const {
    Vector<T,Arch,ct_size+1> res( Reserved{} );
    for( PI i = 0; i < size(); ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    new ( res.data() + size() ) T( value );
    return res;
}

UTPH auto DTP::without_index( PI ind_to_remove ) const {
    Vector<T,Arch,ct_size-1> res( Reserved{} );
    for( PI i = 0; i < ind_to_remove; ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    for( PI i = ind_to_remove + 1; i < size(); ++i )
        new ( res.data() + i - 1 ) T( operator[]( i ) );
    return res;
}

UTPH const T* DTP::data() const {
    return reinterpret_cast<const T *>( this->_storage );
}

UTPH T* DTP::data() {
    return reinterpret_cast<T *>( this->_storage );
}

UTPH auto DTP::begin() const {
    return data();
}

UTPH auto DTP::begin() {
    return data();
}

UTPH auto DTP::end() const {
    return data() + size();
}

UTPH auto DTP::end() {
    return data() + size();
}

UTPH PI DTP::arg_max() const {
    PI res = 0;
    for( PI i = 1; i < size(); ++i )
        if ( operator[]( res ) < operator[]( i ) )
            res = i;
    return res;
}

UTPH T DTP::max() const {
    T res = operator[]( 0 );
    for( PI i = 1; i < size(); ++i )
        if ( res < operator[]( i ) )
            res = i;
    return res;
}

#undef UTPH
#undef UTP
#undef DTP

} // namespace sdot
