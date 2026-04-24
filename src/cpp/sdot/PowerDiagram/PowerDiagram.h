#pragma once

#include "../support/TensorView.h"
#include "../Bsp.h"

namespace sdot {

template<class TF,int ct_dim,class Arch>
struct PowerDiagram {
    //    .pd = PowerDiagram( tensor_view( CtInt<2>(), pd_positions, validity_mask[ 0 ] & 2 ), tensor_view( CtInt<1>(), pd_weights, validity_mask[ 0 ] & 4 ) ),
    TensorView<TF,2,Arch> positions;
    TensorView<TF,1,Arch> weights;
};

}
