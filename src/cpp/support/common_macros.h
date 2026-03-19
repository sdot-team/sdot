#pragma once

#define T_Td template<class T,int d>
#define T_p  template<PI p>
#define T_T  template<class T>

#define ASSERTED_EQUAL( A, B ) ( []( auto a, auto b ) { if ( a != b ) throw std::runtime_error( #A " and " #B " are not equal" ); return a; } )( A, B )
