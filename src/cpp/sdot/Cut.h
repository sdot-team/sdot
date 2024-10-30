#pragma once

#include <tl/support/containers/Vec.h>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TF,int nb_dims,class CutInfo>
class Cut {
public:
    using   Pt    = Vec<TF,nb_dims>;

    CutInfo info;
    Pt      dir;
    TF      off;
};

} // namespace sdot
