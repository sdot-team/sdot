#pragma once

#include "MemorySpace_CpuRam.h" // IWYU pragma: keep  (generic sdot::memory_space)

namespace sdot {

// ---------------------------------------------------------------------------
// Explicit access mode for an operand passed to run_*().
//
// The mode is what decides, when an operand must be transferred to the execution
// space, whether data is copied in before the run and/or back after it:
//   in    : read-only   -> copy in,  no copy back
//   out   : write-only  -> no copy in, copy back
//   inout : read-write  -> copy in,  copy back
// It is orthogonal to the element type's const-ness (a non-const view may still
// be a pure input). An unwrapped operand defaults to read-only (`in`) semantics
// via the operand's own make_accessible().
//
// Each marker simply forwards make_accessible() to the matching variant of the
// wrapped value, so the generic free make_accessible() handles markers unchanged.
// ---------------------------------------------------------------------------

template<class T> struct AccessInp {
    T value;
    HD void make_accessible( const auto &execution_space, auto &&func ) const { value.make_accessible( execution_space, FORWARD( func ) ); }
};
template<class T> struct AccessOut {
    T value;
    HD void make_accessible( const auto &execution_space, auto &&func ) const { value.make_accessible_out( execution_space, FORWARD( func ) ); }
};
template<class T> struct AccessMut {
    T value;
    HD void make_accessible( const auto &execution_space, auto &&func ) const { value.make_accessible_mut( execution_space, FORWARD( func ) ); }
};

// transfer_cost for AccessMode wrappers: delegate to the wrapped value
template<class T> HD auto transfer_cost( const auto &ec, const AccessInp<T> &a ) { return sdot::transfer_cost( ec, a.value ); }
template<class T> HD auto transfer_cost( const auto &ec, const AccessOut<T> &a ) { return sdot::transfer_cost( ec, a.value ); }
template<class T> HD auto transfer_cost( const auto &ec, const AccessMut<T> &a ) { return sdot::transfer_cost( ec, a.value ); }

HD auto inp( auto &&value ) { return AccessInp<DECAYED_TYPE_OF( value )>{ FORWARD( value ) }; }
HD auto out( auto &&value ) { return AccessOut<DECAYED_TYPE_OF( value )>{ FORWARD( value ) }; }
HD auto mut( auto &&value ) { return AccessMut<DECAYED_TYPE_OF( value )>{ FORWARD( value ) }; }

} // namespace sdot
