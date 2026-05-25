#pragma once

#include "../common_macros.h"

namespace sdot {

// ---------------------------------------------------------------------------
// Free-function dispatch.
//
//   run_parallel  ( list, func, args... ); // space inferred from the args
//   run_sequential( ... );                 // == run_parallel with a single thread
//
// The hardware is implied by the data: each space-aware `arg` exposes a
// `memory_kind`. When no execution space is given, it is inferred from the args
// (to start: if any arg's data lives on the GPU, run everything on the GPU).
// Before running, every arg is made accessible from `space` (passed through if
// already reachable, otherwise transferred). `func` is an enriched functor: it
// provides operator()( index, accessible_args... ) and MAY expose resource-hint
// methods (see KernelTraits.h) the launch computation reads.
//
// run_parallel is the primitive: on host it uses the thread pool (all threads),
// on GPU it launches a grid of blocks sized by the launch policy. run_sequential
// forwards to it constraining execution to a single thread (host: 1 worker;
// GPU: a single device thread).
// ---------------------------------------------------------------------------

/// call func for each list item, parallel way.
/// HD: callable from device code too — there it means "already inside a kernel", so it runs
/// inline (sequential on the current thread), the device counterpart of the CPU inline path.
HD void run_parallel( auto &&list, auto &&func, auto &&...args );

/// call func for each list item, one by one
HD void run_sequential( auto &&list, auto &&func, auto &&...args );

} // namespace sdot

#include "Run.cxx" // IWYU pragma: export
