#pragma once

#include "ASSERT.h"
#include "Point.h"

namespace sdot {

#define UTP template<class T,int ct_size,class Arch>
#define DTP Point<T,ct_size,Arch>

UTP T_U DTP::Point( const std::initializer_list<U> &values ) : Point( values.size() ) {
    auto iter = values.begin();
    for( PI i = 0; i < std::min( values.size(), size() ); ++i )
        content[ i ] = *( iter++ );
    for( PI i = values.size(); i < size(); ++i )
        content[ i ] = 0;
}

UTP DTP::Point( const auto &values ) requires( requires { values.size(); } ) : Point( values.size() ) {
    auto iter = values.begin();
    for( PI i = 0; i < std::min( values.size(), size() ); ++i )
        content[ i ] = *( iter++ );
    for( PI i = values.size(); i < size(); ++i )
        content[ i ] = 0;
}

UTP DTP::Point( PI size, auto &&value ) {
    if constexpr ( ct_size < 0 )
        content.resize( size, value );
    else
        ASSERT( size == ct_size );
    for( auto &v : content )
        v = value;
}

UTP DTP::Point( PI size ) {
    if constexpr ( ct_size < 0 )
        content.resize( size );
    else
        ASSERT( size == ct_size );
    for( auto &v : content )
        v = 0;
}

UTP template<class U,std::size_t p> DTP::operator std::array<U,p>() const {
    std::array<U,p> res;
    for( PI i = 0; i < std::min( p, size() ); ++i )
        res[ i ] = content[ i ];
    for( PI i = size(); i < p; ++i )
        res[ i ] = 0;
    return res;
}

UTP template<class U> DTP::operator std::vector<U>() const {
    std::vector<U> res( size() );
    for( PI i = 0; i < size(); ++i )
        res[ i ] = content[ i ];
    return res;
}

UTP DTP::operator std::span<T>() const {
    return std::span<T>( const_cast<T *>( content.data() ), content.size() );
}

UTP const T& DTP::operator[]( PI index ) const {
    return content[ index ];
}

UTP T& DTP::operator[]( PI index ) {
    return content[ index ];
}

UTP bool DTP::operator<( const std::span<T> &that ) const {
    return content < that.data;
}

UTP bool DTP::operator<( const Point &that ) const {
    return content < that.content;
}

UTP auto DTP::with_pushed_value( T value ) const {
    constexpr int new_ct_size = ct_size >= 0 ? ct_size + 1 : -1;
    Point<T,new_ct_size,Arch> res( ct_size + 1 );
    for( PI i = 0; i < size(); ++i )
        res[ i ] = content[ i ];
    res[ size() ] = value;
    return res;
}

UTP auto DTP::without_index( PI ind_to_remove ) const {
    constexpr int new_ct_size = ct_size >= 0 ? ct_size - 1 : -1;
    Point<T,new_ct_size,Arch> res( ct_size - 1 );
    for( PI i = 0; i < ind_to_remove; ++i )
        res[ i ] = content[ i ];
    for( PI i = ind_to_remove + 1; i < size(); ++i )
        res[ i - 1 ] = content[ i ];
    return res;
}

UTP PI DTP::size() const {
    return content.size();
}

UTP const T* DTP::data() const {
    return content.data();
}

UTP T* DTP::data() {
    return content.data();
}

UTP auto DTP::begin() const {
    return content.begin();
}

UTP auto DTP::begin() {
    return content.begin();
}

UTP auto DTP::end() const {
    return content.end();
}

UTP auto DTP::end() {
    return content.end();
}

UTP PI DTP::arg_max() const {
    PI res = 0;
    for( PI i = 1; i < size(); ++i )
        if ( content[ res ] < content[ i ] )
            res = i;
    return res;
}

#undef UTP
#undef DTP

} // namespace sdot

template<class T,int dim,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::Point<T,dim,Arch> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}
