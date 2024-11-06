#pragma once

#include <tl/support/containers/CstSpan.h>
#include <tl/support/containers/Vec.h>
#include <tl/support/TODO.h>
#include <type_traits>
#include <cstring>

namespace sdot {
/**
 * @brief get serialized or deserialized version of a given type
 * 
 * dynamic_mpi_size == true to say that serialization size depends on the value (not compile time known)
 */
template<class T,class B=PI8> struct MpiContent;

//
template<class T,class B> requires TriviallyCopyable<T>
struct MpiContent<T,B> {
public:
    static constexpr bool dynamic_mpi_size = false;
    
    static auto as_cpp( CstSpan<B> mpi_data, auto &&cb ) { 
        return cb( CstSpan<T>( reinterpret_cast<const T *>( mpi_data.data() ), mpi_data.size() / sizeof( T ) ) );
    }
    
    static auto as_mpi( CstSpan<T> cpp_data, auto &&cb ) {
        return cb( CstSpan<B>( reinterpret_cast<const B *>( &cpp_data ), cpp_data.size() * sizeof( T ) ) );
    }
};

//
template<class T,class B> requires TriviallyCopyable<T>
struct MpiContent<Vec<T>,B> {
public:
    static constexpr bool dynamic_mpi_size = true;

    static auto as_cpp( CstSpan<B> mpi_data, CstSpan<PI> mpi_sizes, auto &&cb ) {
        TODO;
    }

    static auto as_cpp( CstSpan<B> mpi_data, PI mpi_size, auto &&cb ) {
        TODO;
    }

    static auto as_mpi( CstSpan<T> cpp_data, auto &&cb ) { 
        TODO;
    }
};

} // namespace sdot
