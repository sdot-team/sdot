#pragma once

#include "../support/DynamicAxis.h"
#include "../geometry/Bsp.h"

namespace sdot {

template<class TF,class Arch,int ct_dim_>
struct PowerDiagram {
    TensorView<TF,2,Arch> positions;
    TensorView<TF,1,Arch> weights;
    CtInt<ct_dim_>        ct_dim;
    Bsp<TF,Arch>          bsp;
};

}
