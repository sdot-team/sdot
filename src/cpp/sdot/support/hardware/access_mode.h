#pragma once

#include "MemorySpace.h" // IWYU pragma: keep  (generic sdot::memory_space)
#include "../common_macros.h"

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

// memory_space() is forwarded (via the generic sdot::memory_space) so the wrapped operand still
// "pulls" toward its native execution space in execution_space_for(), which runs before unwrapping.
template<class T> struct AccessIn {
    T value;
    HD auto memory_space() const requires ( requires { sdot::memory_space( value ); } ) { return sdot::memory_space( value ); }
    HD void make_accessible( const auto &execution_space, auto &&func ) const { value.make_accessible( execution_space, FORWARD( func ) ); }
};
template<class T> struct AccessOut {
    T value;
    HD auto memory_space() const requires ( requires { sdot::memory_space( value ); } ) { return sdot::memory_space( value ); }
    HD void make_accessible( const auto &execution_space, auto &&func ) const { value.make_accessible_out( execution_space, FORWARD( func ) ); }
};
template<class T> struct AccessInOut {
    T value;
    HD auto memory_space() const requires ( requires { sdot::memory_space( value ); } ) { return sdot::memory_space( value ); }
    HD void make_accessible( const auto &execution_space, auto &&func ) const { value.make_accessible_inout( execution_space, FORWARD( func ) ); }
};

HD auto in   ( auto &&value ) { return AccessIn   <DECAYED_TYPE_OF( value )>{ FORWARD( value ) }; }
HD auto out  ( auto &&value ) { return AccessOut  <DECAYED_TYPE_OF( value )>{ FORWARD( value ) }; }
HD auto inout( auto &&value ) { return AccessInOut<DECAYED_TYPE_OF( value )>{ FORWARD( value ) }; }

} // namespace sdot
