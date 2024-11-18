#pragma once

#include "../acceleration_structures/AffineTransformation.h"
#include <tl/support/operators/product.h>
#include "Inst.h"

namespace sdot {

/** */
template<class TF,int nb_dims>
class Img : public Inst {
public:
    using              Tr           = AffineTransformation<TF,nb_dims>;

    virtual void       ct_rt_split  ( CompactReprWriter &cw, Vec<std::pair<const Inst *,ExprData>> &data_map ) const override;
    virtual void       display      ( Displayer &ds ) const override;
    virtual int        type         () const override { return type_Img; }

    PI                 nb_values    () const { return product( extents ); }

    int                interpolation_order;
    Vec<PI,nb_dims>    extents;
    Vec<TF>            values;
    Tr                 trinv;
};

}

#include "Img.cxx" // IWYU pragma: export
