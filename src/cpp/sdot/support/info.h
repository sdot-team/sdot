#pragma once

#include "string/read_arg_name.h"
#include "display.h"
#include <mutex>

namespace sdot {

void __print_with_mutex( std::ostream &os, std::string_view arg_names, const auto &...arg_values ) {
    static std::mutex m;
    m.lock();

    // write
    int cpt = 0;
    auto get_item = [&]( const auto &arg_value ) {
        if ( cpt++ )
            os << "\t";
        display( os << "\033[90m" << read_arg_name( arg_names ) << ":\033[0m ", arg_value );
    };
    ( get_item( arg_values ), ... );
    os << std::endl;

    //
    m.unlock();
}

// PRINT in cout
#define info( ... ) \
    sdot::__print_with_mutex( std::cout, #__VA_ARGS__, __VA_ARGS__ )

} // namespace sdot
