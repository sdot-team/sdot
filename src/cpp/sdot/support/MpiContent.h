#pragma once

#include <tl/support/containers/CstSpan.h>
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
    static constexpr bool dynamic = false; ///< 
    static auto as_cpp( CstSpan<B> mpi, auto &&func ) { return func( CstSpan<T>( reinterpret_cast<const T *>( mpi.data() ), mpi.size() / sizeof( T ) ) ); }
    static auto as_mpi( CstSpan<T> cpp, auto &&func ) { return func( CstSpan<B>( reinterpret_cast<const B *>( cpp.data() ), cpp.size() * sizeof( T ) ) ); }
    
};


} // namespace sdot
