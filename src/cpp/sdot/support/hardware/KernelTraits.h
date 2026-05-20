#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

// ---------------------------------------------------------------------------
// Resource hints for an "enriched" kernel functor.
//
// A functor passed to run() must provide `operator()( index, args... )`. It MAY
// additionally describe its resource usage so the launch computation can pick
// blocks/threads (GPU) or chunking (host) automatically.
//
// Hints are exposed as METHODS (not static members):
//   - they may return a Ct<...> when the value is known at compile time,
//     or a plain runtime value otherwise,
//   - they receive the same arguments passed to the functor, so a hint can
//     depend on the inputs (e.g. required memory scales with an input size).
//     Write `const auto &...` to ignore the arguments.
//
// Two non-exclusive ways to supply hints:
//   1. inherit DefaultKernelTraits and override the methods you care about,
//   2. just define the method(s) directly on your functor.
// Anything left unspecified falls back to the default below, so a plain lambda
// needs to provide nothing.
// ---------------------------------------------------------------------------
namespace default_kernel_traits {
    inline constexpr int nb_gpu_register_per_thread = 32;
    inline constexpr PI  local_gpu_memory_size      = 0;
    inline constexpr PI  local_cpu_memory_size      = 0;
}

struct DefaultKernelTraits {
    auto nb_gpu_register_per_thread( const auto &... ) const { return Ct<int,default_kernel_traits::nb_gpu_register_per_thread>(); }
    auto local_gpu_memory_size     ( const auto &... ) const { return Ct<PI, default_kernel_traits::local_gpu_memory_size>(); }
    auto local_cpu_memory_size     ( const auto &... ) const { return Ct<PI, default_kernel_traits::local_cpu_memory_size>(); }
};

// --- read a trait by calling the functor's method (with the run args) if present, else default ---
template<class F, class... Args>
auto kernel_nb_gpu_register_per_thread( const F &func, const Args &...args ) {
    if constexpr ( requires { func.nb_gpu_register_per_thread( args... ); } ) return func.nb_gpu_register_per_thread( args... );
    else                                                                     return Ct<int,default_kernel_traits::nb_gpu_register_per_thread>();
}

template<class F, class... Args>
auto kernel_local_gpu_memory_size( const F &func, const Args &...args ) {
    if constexpr ( requires { func.local_gpu_memory_size( args... ); } ) return func.local_gpu_memory_size( args... );
    else                                                                 return Ct<PI,default_kernel_traits::local_gpu_memory_size>();
}

template<class F, class... Args>
auto kernel_local_cpu_memory_size( const F &func, const Args &...args ) {
    if constexpr ( requires { func.local_cpu_memory_size( args... ); } ) return func.local_cpu_memory_size( args... );
    else                                                                 return Ct<PI,default_kernel_traits::local_cpu_memory_size>();
}

} // namespace sdot
