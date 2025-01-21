#pragma once

#include "../SymVecValues.h"
#include "Inst.h"

namespace sdot {

/** */
class ValueFromVec : public Inst {
public:
    static RcPtr<Inst>  from_vec      ( const RcPtr<SymVecValues> &vec, const RcPtr<Inst> &ind );
 
    virtual bool        always_equal  ( const Inst &that ) const override;
    virtual void        ct_rt_split   ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual void        display       ( Displayer &ds ) const override;
    virtual int         type          () const override { return type_ValueFromVec; }
    virtual RcPtr<Inst> subs          ( const std::map<Str,RcPtr<Inst>> &map ) const override;
     
    RcPtr<SymVecValues> vec;
};

}
