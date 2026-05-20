#pragma once

#include "../common_macros.h"
#include "MemorySpace.h" // IWYU pragma: export

namespace sdot {

// ---------------------------------------------------------------------------
// "Informed" pointer: a raw address tagged at compile time with the kind of
// memory it addresses. Same size and overhead as a raw pointer — the Kind tag
// is purely compile-time, the runtime device identity (device_id/stream) lives
// in the MemorySpace instance passed to run()/arch_copy(), not here.
//
// Deref is intentionally NOT host/device-guarded: a space mismatch is resolved
// by a transfer at the operation layer (run/get_data_from/...), not by a
// compile error. The Kind tag drives transfer-primitive selection and
// accessibility queries.
// ---------------------------------------------------------------------------
template<class T, class _MemorySpace>
struct Ptr {
    using            MemorySpace  = _MemorySpace;
    using            value_type   = T;

    HD explicit      Ptr          ( T *raw = nullptr, MemorySpace memory_space = {} ) : memory_space( memory_space ), raw( raw ) {}

    // byte-offset arithmetic (T is expected to be std::byte / const std::byte for strided views)
    HD Ptr           operator+    ( auto off ) const { return Ptr( raw + off, memory_space ); }
    HD Ptr           operator-    ( auto off ) const { return Ptr( raw - off, memory_space ); }

    // reinterpret to another element type, keeping the same memory kind
    template<class U> HD U *as    () const { return reinterpret_cast<U *>( raw ); }

    HD explicit      operator bool() const { return raw != nullptr; }
    HD bool          operator==   ( const Ptr &o ) const { return memory_space == o.memory_space && raw == o.raw; }
    HD bool          operator!=   ( const Ptr &o ) const { return ! operator==( o ); }

    MemorySpace      memory_space; ///< void structure by default
    T*               raw;
};

} // namespace sdot
