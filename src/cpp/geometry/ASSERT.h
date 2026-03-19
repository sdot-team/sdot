#pragma once

#include <iostream>
#include <assert.h>

#ifdef TL_DEBUG
    #define ASSERT_IF_DEBUG( COND ) assert( COND );
#else
    #define ASSERT_IF_DEBUG( COND )
#endif

auto __asserted( auto &&value, auto file, auto line, auto str ) {
    if ( ! value ) {
        std::cerr << file << ":" << line << ": condition not met: " << str << ";";
        assert( 0 );
    }
    return std::forward<decltype( value )>( value );
}

#define ASSERTED_POSITIVE( VALUE ) ( [&]( auto &&v ) { assert( v >= 0 ); return v; } )( VALUE )
#define ASSERTED( VALUE ) __asserted( VALUE, __FILE__, __LINE__, #VALUE )
#define ASSERT( COND ) ( [&]( bool v ) { if ( v ) return; std::cerr << __FILE__ << ":" << __LINE__ << ": condition not met: " #COND ";"; assert( 0 ); } )( bool( COND ) )
