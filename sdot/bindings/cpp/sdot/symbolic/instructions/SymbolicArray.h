#pragma once

#include <tl/support/operators/product.h>
#include "Inst.h"

namespace sdot {

/**
    Stores array data + symbolic children representing the indices

    Beware : this is a raw symbolic array, there's no index checking
*/
template<class TF,int nb_dims>
class SymbolicArray : public Inst {
public:
    virtual int         compare_same ( const Inst &that ) const override;
    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info    () const override;
    virtual void        display      ( Displayer &ds ) const override;
    virtual RcPtr<Inst> clone        ( Vec<RcPtr<Inst>> &&new_children ) const override;
    virtual int         type         () const override { return type_Array; }

    PI                  nb_values    () const { return product( extents ); }
 
    Vec<PI,nb_dims>     extents;
    Vec<TF>             values;
};

}

#include "SymbolicArray.cxx" // IWYU pragma: export
