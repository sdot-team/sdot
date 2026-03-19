#pragma once

#include "PointFactory.h"

namespace sdot {

#define UTP template<class T,int dim>
#define DTP PointFactory<T,dim>

#undef UTP
#undef DTP

// --------------------------------------------------------------------------------------
#define UTP template<class T>
#define DTP PointFactory<T,-1>

#undef UTP
#undef DTP

} // namespace sdot

