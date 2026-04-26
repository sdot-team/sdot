#include "../support/DynamicAxis.h"
#include "../support/P.h"
using namespace sdot;

template<class TF,class PI,class Arch>
struct Yo {
    TensorView<TF,2,Arch> a;
    TensorView<PI,0,Arch> nb_points;
};

template<class TF,class Arch>
struct BatchOfYo {
    TensorView<TF,3,Arch> a;
    DynamicAxis<1,Arch> nb_points;
};

void yo( auto &&p ) {
    p.ret.nb_points( 0 ) = 2;
}

void yo_backward( auto && ) {
}
