#pragma once

#include "PointFactory.h"

namespace sdot {

#define UTP template<class T,int ct_dim,class Arch>
#define DTP PointFactory<T,ct_dim,Arch>

#undef UTP
#undef DTP

// --------------------------------------------------------------------------------------
#define UTP template<class T,class Arch>
#define DTP PointFactory<T,-1,Arch>

#undef UTP
#undef DTP

} // namespace sdot

