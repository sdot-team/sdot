#pragma once

#include "../Ct.h"
#include "Tuple.h"

namespace sdot {

/// Compile-time-zero tensor: every element reads as TF(0).
///   Shape must satisfy the same constraints as TensorView's shape (a Tuple,
///   with Ct<TI,V> entries for compile-time-known axes).
///   No MemorySpace — zero tensors are accessible from any execution context
///   without transfer (transfer_cost always returns 0_c).
template<class _TF, class _Shape>
class ZeroTensor {
public:
    using            TF                   = _TF;
    using            Shape                = _Shape;
    using            TI                   = SI;

    SCInt            ct_rank              = Shape::ct_size;

    HD               ZeroTensor           ( Shape shape );
    HD               ZeroTensor           () = default;

    // status — returns Ct<bool,...> so callers can use if constexpr
    HD auto          surely_null          () const { return 1_b; }
    HD auto          is_invalid           () const { return 0_b; }
    HD auto          is_valid             () const { return 1_b; }

    // shape
    HD Shape         shape                () const { return _shape; }
    HD auto          shape                ( auto d ) const { return _shape[ d ]; }
    HD auto          nb_items             () const;

    // always accessible from any execution context — no transfer needed
    HD auto          transfer_cost        ( const auto & ) const { return 0_c; }

    // indexing — peels one dimension per call, result is always zero
    HD auto          operator()           () const { return *this; }
    HD auto          operator()           ( auto index, auto ...rem ) const;
    HD auto          operator[]           ( auto index ) const { return operator()( index ); }

    // scalar access
    HD               operator TF          () const { return TF(0); }
    HD TF            value                () const { return TF(0); }

private:
    Shape            _shape;
};

} // namespace sdot

#include "ZeroTensor.cxx" // IWYU pragma: export
