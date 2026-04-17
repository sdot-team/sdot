#pragma once

#include <vector>
#include "../support/P.h"

namespace sdot {

template<class TF, class Arch>
struct CutWorkspace {
    std::vector<TF> sps;
    std::vector<PI> indices_to_remove;  ///< reused for edges, vertices, cuts
    std::vector<PI> corr;               ///< reused for vertex_corr, cut_corr
    std::vector<bool> used_flags;
};

} // namespace sdot
