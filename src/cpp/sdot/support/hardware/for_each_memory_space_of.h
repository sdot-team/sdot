#pragma once

namespace sdot {

/// func( memory_space, callback ), callback() must return nb_bytes for this memory space.
/// func can be called several times for the same memory_space and different values
auto for_each_memory_space_of( const auto &, auto &&/*func*/ ) {
}

} // namespace sdot
