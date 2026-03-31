#pragma once

#define T_TdA template<class T,int d,class A>
#define T_Up  template<class U,std::size_t p>
#define T_TA  template<class T,class A>
#define T_Td  template<class T,int d>
#define T_T   template<class T>
#define T_U   template<class U>
#define T_d   template<int d>
#define T_p   template<PI p>

#define ASSERTED_EQUAL( A, B ) ( []( auto a, auto b ) { if ( a != b ) throw std::runtime_error( #A " and " #B " are not equal" ); return a; } )( A, B )

#ifdef __CUDACC__
#  define HD __host__ __device__
#else
#  define HD
#endif
