#pragma once

#include "../support/DsVec.h"

namespace sdot {

template<int dim,int npt,class TF,class Arch>
struct Simplex {
    using Pt = DsVec<TF,dim,Arch>;
    std::array<Pt,npt> pts;
};

} // namespace sdot
