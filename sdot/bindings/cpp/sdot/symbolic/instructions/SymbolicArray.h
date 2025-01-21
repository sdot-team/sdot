#pragma once

#include <tl/support/operators/product.h>
#include "Inst.h"

namespace sdot {

/** */
template<class TF,int nb_dims>
class SymbolicArray : public Inst {
public:
    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual void        display      ( Displayer &ds ) const override;
    virtual int         type         () const override { return type_Array; }
    virtual RcPtr<Inst> subs         ( const std::map<Str,RcPtr<Inst>> &map ) const override;

    PI                  nb_values    () const { return product( extents ); }
 
    Vec<PI,nb_dims>     extents;
    Vec<TF>             values;
};

}

#include "SymbolicArray.cxx" // IWYU pragma: export
