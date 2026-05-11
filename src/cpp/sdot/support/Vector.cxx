#pragma once

#include <algorithm>
#include <utility>
#include "Ct.h"

#include "Vector.h"
#include "CtdInt.h"

namespace sdot {

#define UTP template<class T,class Arch,int ct_size>
#define DTP Vector<T,Arch,ct_size>

UTP DTP::Vector( const auto &values ) requires( requires { values.size(); } ) : Vector( Reserved(), values.size() ) {
    ASSERT( values.size() == size() );
    auto iter = values.begin();
    for( PI i = 0; i < size(); ++i )
        new ( data() + i ) T( *( iter++ ) );
}

UTP DTP::Vector( Reserved, PI size ) {
    if constexpr ( ct_size < 0 ) {
        content.values = reinterpret_cast<T * >( malloc( size * sizeof( T ) ) );
        content.size = size;
    } else {
        ASSERT_EQ( size, ct_size );
    }
}

UTP DTP::Vector( Size, PI size, auto &&...ctor_args ) : Vector( Reserved(), size ) {
    for( auto &v : *this )
        new ( &v ) T( ctor_args... );
}

UTP DTP::Vector() {
    static_assert( ct_size >= 0 );
    for( auto &v : *this )
        new ( &v ) T;
}

UTP DTP::Vector( Values, auto &&...values ) : Vector( Reserved(), sizeof...( values ) ) {
    PI i = 0;
    auto append = [&]( auto &&value ) {
        new ( data() + i++ ) T( FORWARD( value ) );
    };
    ( append( FORWARD( values ) ), ... );
}

UTP DTP::Vector( const Vector &that ) : Vector( Reserved(), that.size() ) {
    for ( PI i = 0; i < that.size(); ++i )
        new ( data() + i ) T( that[ i ] );
}

UTP DTP::Vector( Vector &&that ) noexcept {
    if constexpr ( ct_size < 0 ) {
        content.values = std::exchange( that.content.values, nullptr );
        content.size = std::exchange( that.content.size, 0 );
    } else {
        for ( PI i = 0; i < PI( ct_size ); ++i )
            new ( data() + i ) T( std::move( that[ i ] ) );
    }
}

UTP Vector<T,Arch,ct_size>& DTP::operator=( const Vector &that ) {
    if ( this == &that )
        return *this;

    if constexpr ( ct_size < 0 ) {
        // if not possible to reuse the allocation
        if ( content.size != that.content.size ) {
            for ( PI i = 0; i < content.size; ++i )
                data()[ i ].~T();
            if ( content.values )
                free( content.values );

            content.values = reinterpret_cast<T *>( malloc( that.content.size * sizeof( T ) ) );
            content.size = that.content.size;
            for ( PI i = 0; i < content.size; ++i )
                new ( data() + i ) T( that[ i ] );

            return *this;
        }
    }

    for ( PI i = 0; i < size(); ++i )
        operator[]( i ) = that[ i ];

    return *this;
}

UTP DTP& DTP::operator=( Vector &&that ) noexcept {
    if ( this == &that )
        return *this;

    if constexpr ( ct_size < 0 ) {
        for ( PI i = 0; i < content.size; ++i )
            data()[ i ].~T();
        if ( content.values )
            free( content.values );
        content.values = std::exchange( that.content.values, nullptr );
        content.size = std::exchange( that.content.size, 0 );
    } else {
        for ( PI i = 0; i < PI( ct_size ); ++i )
            operator[]( i ) = std::move( that[ i ] );
    }

    return *this;
}

UTP DTP::~Vector() {
    for ( PI i = 0; i < size(); ++i )
        data()[ i ].~T();
    if constexpr ( ct_size < 0 )
        if ( content.values )
            free( content.values );
}

UTP template<class U,std::size_t p> DTP::operator std::array<U,p>() const {
    std::array<U,p> res;
    for( PI i = 0; i < std::min( p, size() ); ++i )
        res[ i ] = operator[]( i );
    for( PI i = size(); i < p; ++i )
        res[ i ] = 0;
    return res;
}

UTP template<class U> DTP::operator std::vector<U>() const {
    std::vector<U> res( size() );
    for( PI i = 0; i < size(); ++i )
        res[ i ] = operator[]( i );
    return res;
}

UTP DTP::operator std::span<T>() const {
    return std::span<T>( const_cast<T *>( data() ), size() );
}

UTP const T& DTP::operator[]( PI index ) const {
    return data()[ index ];
}

UTP T& DTP::operator[]( PI index ) {
    return data()[ index ];
}

UTP bool DTP::operator<( const std::span<T> &that ) const {
    return std::ranges::lexicographical_compare( operator std::span<T>(), that );
}

UTP bool DTP::operator<( const Vector &that ) const {
    return std::ranges::lexicographical_compare( operator std::span<T>(), that.operator std::span<T>() );
}

UTP DTP DTP::with_func( PI size, auto &&func ) {
    Vector res( Reserved(), size );
    for ( PI i = 0; i < size; ++i )
        new ( res.data() + i ) T( func( i ) );
    return res;
}

UTP DTP DTP::zeros( PI size ) {
    return with_func( size, []( PI ) { return T( 0 ); } );
}

UTP DTP DTP::ones( PI size ) {
    return with_func( size, []( PI ) { return T( 1 ); } );
}

UTP DTP DTP::value_at( PI size, PI index, T value ) {
    return with_func( size, [&]( PI i ) { return i == index ? value : T( 0 ); } );
}

UTP auto DTP::with_pushed_value( T value ) const {
    Vector<T,Arch,ctd_add( ct_size, 1 )> res( Reserved(), size() + 1 );
    for( PI i = 0; i < size(); ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    new ( res.data() + size() ) T( value );
    return res;
}

UTP auto DTP::without_index( PI ind_to_remove ) const {
    Vector<T,Arch,ctd_sub( ct_size, 1 )> res( Reserved(), size() - 1 );
    for( PI i = 0; i < ind_to_remove; ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    for( PI i = ind_to_remove + 1; i < size(); ++i )
        new ( res.data() + i - 1 ) T( operator[]( i ) );
    return res;
}

UTP auto DTP::size() const {
    if constexpr ( ct_size < 0 )
        return content.size;
    else
        return Ct<PI,ct_size>();
}

UTP const T* DTP::data() const {
    return reinterpret_cast<const T *>( content.values );
}

UTP T* DTP::data() {
    return reinterpret_cast<T *>( content.values );
}

UTP auto DTP::begin() const {
    return data();
}

UTP auto DTP::begin() {
    return data();
}

UTP auto DTP::end() const {
    return data() + size();
}

UTP auto DTP::end() {
    return data() + size();
}

UTP PI DTP::arg_max() const {
    PI res = 0;
    for( PI i = 1; i < size(); ++i )
        if ( operator[]( res ) < operator[]( i ) )
            res = i;
    return res;
}

UTP T DTP::max() const {
    T res = operator[]( 0 );
    for( PI i = 1; i < size(); ++i )
        if ( res < operator[]( i ) )
            res = i;
    return res;
}

#undef UTP
#undef DTP

} // namespace sdot
