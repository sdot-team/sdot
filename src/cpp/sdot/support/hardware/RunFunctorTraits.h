#pragma once

#include "../common_macros.h"
#include <thread>

namespace sdot {

namespace RunFunctorTraits {

int max_cpu_threads( auto &&func, auto &&...args ) {
    if constexpr( requires{ func.max_cpu_threads( args... ); } )
        return func.max_cpu_threads( args... );
    else
        return std::thread::hardware_concurrency();
}

void per_thread( auto &func, auto &&thread_info, auto &&list, auto &&cont, auto &&...args ) {
    if constexpr( requires{ func.per_thread( FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... ); } )
        func.per_thread( FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... );
    else
        cont( FORWARD( args )... ); //
}


///
template<class Func>
struct RunFunctorWrapper {
    int  max_cpu_threads( auto &&...args ) { return sdot::RunFunctorTraits::max_cpu_threads( func, FORWARD( args )... ); }
    void per_thread     ( auto &&thread_info, auto &&list, auto &&cont, auto &&...args ) { return sdot::RunFunctorTraits::per_thread( func, FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... ); }
    auto operator()     ( auto &&...args ) const { func( FORWARD( args )... ); }

    Func func;
};

} // RunFunctorTraits

} // namespace sdot
