#pragma once

namespace sdot {

// ---------------------------------------------------------------------------
// Free-function dispatch.
//
//   run_parallel  ( batch_sizes, func, args... );          // space inferred from the args
//   run_parallel  ( space, batch_sizes, func, args... );   // explicit space (e.g. XLA stream)
//   run_sequential( ... );                                 // == run_parallel with a single thread
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

// internal dispatch (defined in Run.cxx): max_nb_threads < 0 means "all threads".
// template<class Space, class BatchSizes, class F, class... Args>
//     requires ExecutionSpace<Space>
// void _run( const Space &space, int max_nb_threads, const BatchSizes &batch_sizes, F &&func, Args &&...args );

// // --- run_parallel ---
// template<class Space, class BatchSizes, class F, class... Args>
//     requires ExecutionSpace<Space>
// void run_parallel( const Space &space, const BatchSizes &batch_sizes, F &&func, Args &&...args ) {
//     _run( space, -1, batch_sizes, std::forward<F>( func ), std::forward<Args>( args )... );
// }

/// call func for each batch value, parallel way
void run_parallel( const auto &batch_sizes, auto &&func, auto &&...args );

/// call func for each batch value, one by one
void run_sequential( const auto &batch_sizes, auto &&func, auto &&...args );

} // namespace sdot

#include "Run.cxx" // IWYU pragma: export
