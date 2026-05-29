#pragma once

#include "../common_macros.h"
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
HD auto transfer_cost( const auto &/*ec*/ ) { return Ct<int,0>{}; }

// single data-arg default: scalars and plain value types carry no heap data
HD auto transfer_cost( const auto &ec, const auto &arg ) {
    if ( requires { arg.transfer_cost( ec ); } )
        return 0_c;
    else
        static_assert( "don't know how to make transfer_cost" );
}

// multi data-args: recursive sum — requires at least 2 to avoid ambiguity with single-arg default
HD auto transfer_cost( const auto &ec, const auto &head, const auto &...tail ) requires ( sizeof...( tail ) >= 1 ) {
    return transfer_cost( ec, head ) + transfer_cost( ec, tail... );
}

} // namespace sdot
