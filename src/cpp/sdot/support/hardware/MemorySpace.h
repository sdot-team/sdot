#pragma once

#include "../common_macros.h"

namespace sdot {

// ---------------------------------------------------------------------------
// MemorySpace = WHAT KIND of memory an address refers to,
// not which execution context.
//
// Their job:
//   - drive transfer-primitive selection (see CrossArchCopy.h / arch_copy),
//   - answer host/device accessibility queries,
//   - document intent and put responsibilities at the pointer level.
// ---------------------------------------------------------------------------
struct MemorySpace {};

// SFINAE via trailing return: only participates when value has a .memory_space() method
T_T auto memory_space( const T &value ) -> decltype( value.memory_space() ) {
    return value.memory_space();
}

} // namespace sdot
