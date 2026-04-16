#pragma once

#include "../support/common_types.h"
#include <vector>

namespace sdot {

/// A facet of a Cell: the (dim-1)-face defined by a single cut.
/// ``vertex_indices`` are indices into the parent cell's vertex arrays.
struct Facet {
    SI              defining_cut;
    std::vector<PI> vertex_indices;
};

} // namespace sdot
