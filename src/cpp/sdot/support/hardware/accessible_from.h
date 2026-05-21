#pragma once

#include "../Ct.h"

namespace sdot {

// ---------------------------------------------------------------------------
// accessible_from( execution_space, memory_space ) -> Ct<bool,...>
//
// Compile-time accessibility matrix: can code running in `execution_space` read/write
// bytes living in `memory_space` without a transfer? Default is "no"; each accessible pair
// is declared in the matching MemorySpace_*.h (e.g. (Cpu,CpuRam), (Cuda,GlobalCudaRam),
// (Cpu|Cuda, PinnedCpuRam)).
//
// This is the single source of truth for both run_* dispatch (make_accessible: pass-through
// vs transfer) and the host/device context static_asserts in TensorView.
// ---------------------------------------------------------------------------
auto accessible_from( const auto &/*execution_space*/, const auto &/*memory_space*/ ) {
    return Ct<bool,false>();
}

} // namespace sdot
