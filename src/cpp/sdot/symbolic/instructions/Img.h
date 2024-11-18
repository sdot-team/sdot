#pragma once

#include "../../acceleration_structures/AffineTransformation.h"
#include <tl/support/operators/product.h>
#include "Inst.h"

namespace sdot {

/** */
template<class TF,int nb_dims,int interpolation_order>
class Img : public Inst {
public:
    using               Tr           = AffineTransformation<TF,nb_dims>;
 
    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual void        display      ( Displayer &ds ) const override;
    virtual int         type         () const override { return type_Img; }
    virtual RcPtr<Inst> subs         ( const std::map<Str,RcPtr<Inst>> &map ) const override;

    PI                  nb_values    () const { return product( extents ); }
 
    Vec<PI,nb_dims>     extents;
    Vec<TF>             values;
    Tr                  trinv;
};

}

#include "Img.cxx" // IWYU pragma: export
