#pragma once

namespace sdot {

// ---------------------------------------------------------------------------
// An execution space is where code runs together with the resource it is bound
// to (host, or a CUDA device + stream). It is distinct from a memory KIND (which
// lives on the Ptr and says where bytes reside): the compile-time kind says
// host-vs-GPU, the execution space adds the runtime identity (device_id, stream)
// that light pointers deliberately do not carry.
//
// Execution spaces are passed (optionally) to run_*(). The "main" one is
// {}-constructible. When omitted, run_*() infers the space from where the
// arguments' data reside.
// ---------------------------------------------------------------------------
struct ExecutionContext {
};

} // namespace sdot

