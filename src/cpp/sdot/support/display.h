#pragma once

#include <iostream>

namespace sdot {

void display( std::ostream &os, const auto &value ) {
    if constexpr ( requires { value.display( os ); } ) {
        value.display( os );
    } else if constexpr ( requires { os << value; } ) {
        os << value;
    } else if constexpr ( requires { DECAYED_TYPE_OF( value.shape().size() )::value; } ) {
        constexpr int rank = DECAYED_TYPE_OF( value.shape.size() )::value;
        const auto shape = value.shape();

        if constexpr ( rank == 0 ) {
            os << value.item();
        } else if constexpr ( rank == 1 ) {
            for( std::size_t i = 0; i < shape[ 0 ]; ++i )
                display( os << ( i ? ", " : "" ), value[ i ] );
        } else {
            for( std::size_t i = 0; i < shape[ 0 ]; ++i )
                display( os << "\n  ", value( i ) );
        }
    } else if constexpr ( requires { std::begin( value ); std::end( value ); } ) {
        auto iter = std::begin( value );
        if ( iter != std::end( value ) ) {
            os << "[ " << *iter;
            while( ++iter != std::end( value ) )
                os << ", " << *iter;
            os << " ]";
        } else
            os << "[]";
    } else {
        os << "TODO: display of " << typeid( value ).name();
    }
}

} // namespace sdot
