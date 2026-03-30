#pragma once

#include "read_arg_name.h"
#include <iostream>
#include <ostream>
#include <vector>
#include <mutex>

template<class T,class A> std::ostream &operator<<( std::ostream &os, const std::vector<T,A> &v ) {
    for( std::size_t i = 0; i < v.size(); ++i )
        os << ( i ? ", " : " " ) << v[ i ];
    return os;
}

namespace uot {

void __print_with_mutex( std::ostream &os, std::string_view arg_names, const auto &...arg_values ) {
    static std::mutex m;
    m.lock();

    // write
    int cpt = 0;
    auto get_item = [&]( const auto &arg_value ) {
        if ( cpt++ )
            os << "\t";
        os << "\033[90m" <<  read_arg_name( arg_names ) << ":\033[0m " << arg_value;
    };
    ( get_item( arg_values ), ... );
    os << std::endl;

    //
    m.unlock();
}

// PRINT in cout
#define P( ... ) \
    uot::__print_with_mutex( std::cout, #__VA_ARGS__, __VA_ARGS__ )

} // namespace uot
