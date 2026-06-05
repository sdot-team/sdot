#pragma once

#include "../Ct.h"

namespace sdot {

/// Absent / undefined tensor (replaces TensorView::make_invalid for the JAX FFI path).
///   Carries only the element type TF — no shape, no data, no MemorySpace.
///   transfer_cost returns 0_c; is_valid returns 0_b so callers can gate on
///   if constexpr ( DECAYED_TYPE_OF( t.is_valid() )::value ).
template<class _TF>
class NoneTensor {
public:
    using TF = _TF;
    using TI = SI;

    SCInt ct_rank = -1; ///< undefined rank (not a real tensor)

    HD auto surely_null  () const { return 1_b; }
    HD auto is_invalid   () const { return 1_b; }
    HD auto is_valid     () const { return 0_b; }

    // always accessible — no transfer needed
    T_U HD auto transfer_cost( const U & ) const { return 0_c; }

    // shape stubs — only reachable in dead branches (is_valid() == 0_b)
    // but must compile for ternary patterns like `t.is_valid() ? t.shape(d) : -1`
    T_U HD SI   shape        ( U ) const { return 0; }
    HD SI   nb_items     () const { return 0; }

    // indexing — returns *this so that kernel code indexing into absent gradients stays compilable
    T_VA HD const NoneTensor& operator()  ( A... ) const { return *this; }
    T_U  HD const NoneTensor& operator[]  ( U )    const { return *this; }

    // scalar conversion — allows `const TF g = none_tensor(i,j)` in dead branches
    HD operator TF() const { return TF(0); }
};

} // namespace sdot
