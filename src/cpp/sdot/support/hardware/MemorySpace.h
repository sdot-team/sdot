#pragma once

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

} // namespace sdot
