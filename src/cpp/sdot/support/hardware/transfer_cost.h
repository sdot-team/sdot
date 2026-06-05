#pragma once

#include "../common_macros.h"
#include "apply_values.h"
#include "../Ct.h"

namespace sdot {

// ---------------------------------------------------------------------------
// transfer_cost( ec, args... ) -> Ct<int, N>
//
// Returns a compile-time cost representing how much data must be moved to make
// all `args` accessible from execution context `ec`:
//   N = 0 : data is already native to that context — no transfer.
//   N > 0 : relative transfer cost (higher = more expensive).
//
// The return TYPE encodes N so the function is only used in unevaluated
// contexts (DECAYED_TYPE_OF / decltype) — zero runtime overhead.
//
// Overload set:
//   1. Zero data-args (ec only)   → Ct<int,0>   (identity for recursive sums)
//   2. Single data-arg (ec + arg) → Ct<int,0>   default (scalars, value types)
//                                   overloads in MemorySpace_*.h and TensorView
//   3. Multi data-args (ec + 2+)  → sum over all args (recursive)
//
// Extension points:
//   Per-memory-space : MemorySpace_CpuRam.h, MemorySpace_GlobalCudaRam.h, …
//   Per-container   : TensorView.cxx (delegates to its MemorySpace)
//   Aggregates      : define an overload summing member costs (fully opt-in)
// ---------------------------------------------------------------------------

// zero data-args: identity
T_T HD auto transfer_cost( const T &/*ec*/ ) { return 0_c; }

T_T HD auto transfer_cost( const T &/*ec*/, Inp ) { return 0_c; }
T_T HD auto transfer_cost( const T &/*ec*/, Out ) { return 0_c; }
T_T HD auto transfer_cost( const T &/*ec*/, Mut ) { return 0_c; }

struct AnyFunctor {
    T_VA HD void operator()( A &&... ) const {}
};

// member-call probes for the single-arg dispatch below
namespace detail {
    template<class Arg,class EC> using m_transfer_cost = decltype( std::declval<Arg>().transfer_cost( std::declval<EC>() ) );
    template<class Arg>          using m_apply_values  = decltype( apply_values( std::declval<Arg&>(), AnyFunctor{} ) );
}

// single data-arg: delegate to the arg's transfer_cost method (required — static_assert if absent)
T_TA HD auto transfer_cost( const T &ec, const A &arg ) {
    if constexpr ( IS_DETECTED( detail::m_transfer_cost, A, T ) )
        return arg.transfer_cost( ec );
    else if constexpr ( std::is_trivial<DECAYED_TYPE_OF( arg )>::value )
        return 0_c;
    else if constexpr ( IS_DETECTED( detail::m_apply_values, A ) )
        return apply_values( arg, [&]( auto &&...values ) { return ( transfer_cost( ec, values ) + ... + 0_c ); } );
    else {
        // static_assert( sizeof( arg ) == 0, "don't know how to make transfer_cost: arg must provide transfer_cost( ec )" );
        arg.no_transfer_cost();
        return 0_c;
    }
}

// multi data-args: recursive sum — enable_if at least 2 to avoid ambiguity with single-arg default
template<class EC,class H,class... A,class=std::enable_if_t<( sizeof...( A ) >= 1 )>>
HD auto transfer_cost( const EC &ec, const H &head, const A &...tail ) {
    return transfer_cost( ec, head ) + transfer_cost( ec, tail... );
}

template<class EC,class... A> auto accessible_from( const EC &ec, const A &...values ) {
    return transfer_cost( ec, values... ) == 0_c;
}

} // namespace sdot
