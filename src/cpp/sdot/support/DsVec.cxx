#pragma once

#include "ASSERT.h"
#include "DsVec.h"
#include <utility>

namespace sdot {

#define UTP template<class T,int ct_size,class Arch>
#define DTP DsVec<T,ct_size,Arch>

UTP DTP::DsVec( const auto &values ) requires( requires { values.size(); } ) : DsVec( Reserved(), values.size() ) {
    ASSERT( values.size() == size() );
    auto iter = values.begin();
    for( PI i = 0; i < size(); ++i )
        new ( data() + i ) T( *( iter++ ) );
}

UTP DTP::DsVec( Reserved, PI size ) {
    if constexpr ( ct_size < 0 ) {
        content.values = reinterpret_cast<T * >( malloc( size * sizeof( T ) ) );
        content.size = size;
    } else {
        ASSERT_EQ( size, ct_size );
    }
}

UTP DTP::DsVec( Size, PI size, auto &&...ctor_args ) : DsVec( Reserved(), size ) {
    for( auto &v : *this )
        new ( &v ) T( ctor_args... );
}

UTP DTP::DsVec( Values, auto &&...values ) : DsVec( Reserved(), sizeof...( values ) ) {
    PI i = 0;
    auto append = [&]( auto &&value ) {
        new ( data() + i++ ) T( FORWARD( value ) );
    };
    ( append( FORWARD( values ) ), ... );
}

UTP DTP::DsVec( const DsVec &that ) : DsVec( Reserved(), that.size() ) {
    for ( PI i = 0; i < that.size(); ++i )
        new ( data() + i ) T( that[ i ] );
}

UTP DTP::DsVec( DsVec &&that ) noexcept {
    if constexpr ( ct_size < 0 ) {
        content.values = std::exchange( that.content.values, nullptr );
        content.size = std::exchange( that.content.size, 0 );
    } else {
        for ( PI i = 0; i < PI( ct_size ); ++i )
            new ( data() + i ) T( std::move( that[ i ] ) );
    }
}

UTP DsVec<T,ct_size,Arch>& DTP::operator=( const DsVec &that ) {
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

UTP DsVec<T,ct_size,Arch>& DTP::operator=( DsVec &&that ) noexcept {
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

UTP DTP::~DsVec() {
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
    return std::span<T>( const_cast<T *>( content.data() ), content.size() );
}

UTP const T& DTP::operator[]( PI index ) const {
    return data()[ index ];
}

UTP T& DTP::operator[]( PI index ) {
    return data()[ index ];
}

UTP bool DTP::operator<( const std::span<T> &that ) const {
    return operator std::span<T>() < that.data;
}

UTP bool DTP::operator<( const DsVec &that ) const {
    return operator std::span<T>() < that.content.operator std::span<T>();
}

UTP DTP DTP::with_func( PI size, auto &&func ) {
    DsVec res( Reserved(), size );
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
    constexpr int new_ct_size = ct_size >= 0 ? ct_size + 1 : -1;
    DsVec<T,new_ct_size,Arch> res( Reserved(), size() + 1 );
    for( PI i = 0; i < size(); ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    new ( res.data() + size() ) T( value );
    return res;
}

UTP auto DTP::without_index( PI ind_to_remove ) const {
    constexpr int new_ct_size = ct_size >= 0 ? ct_size - 1 : -1;
    DsVec<T,new_ct_size,Arch> res( Reserved(), size() - 1 );
    for( PI i = 0; i < ind_to_remove; ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    for( PI i = ind_to_remove + 1; i < size(); ++i )
        new ( res.data() + i - 1 ) T( operator[]( i ) );
    return res;
}

UTP PI DTP::size() const {
    if constexpr ( ct_size < 0 )
        return content.size;
    else
        return ct_size;
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

#undef UTP
#undef DTP

} // namespace sdot

template<class T,int dim,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::DsVec<T,dim,Arch> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}
