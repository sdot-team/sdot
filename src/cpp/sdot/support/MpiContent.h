tk#pragma once

#include <tl/support/containers/CstSpan.h>
#include <tl/support/containers/Vec.h>
#include <type_traits>
#include <cstring>

namespace sdot {
/**
 * @brief get serialized or deserialized version of a given type
 * 
 * dynamic == true to say that size of 
 */
template<class T,class B=PI8> struct MpiContent;

template<class T,class B> requires std::is_trivially_copyable_v<T>
struct MpiContent<T,B> {
public:
    static auto as_cpp( CstSpan<B> mpi, auto size, auto &&cb ) { return cb( *reinterpret_cast<const T *>( mpi.data() ) ); }
    static auto as_mpi( const T &cpp, auto &&cb ) { return cb( CstSpan<B>( reinterpret_cast<const B *>( &cpp ), sizeof( T ) ) ); }
    static auto size  ( const T &cpp ) { return CtInt<sizeof( T )>{}; } ///< size of an individual element
};

template<class T,class B> requires std::is_trivially_copyable_v<T>
struct MpiContent<Vec<T>,B> {
public:
    static auto as_cpp( CstSpan<B> mpi, auto sizes, auto &&cb ) { return cb( *reinterpret_cast<const T *>( mpi.data() ) ); }
    static auto as_mpi( CstSpan<T> cpp, auto &&cb ) { return cb( CstSpan<B>( reinterpret_cast<const B *>( &cpp ), sizeof( T ) ) ); }
    static auto size  ( const T &cpp ) { return CtInt<sizeof( T )>{}; } ///< size of an individual element
};

} // namespace sdot
