#pragma once

#include "FuncWithCoefficients.h"

namespace sdot {

/** */
class Mul : public FuncWithCoefficients {
public:
    static RcPtr<Inst>  from_operands( const Str &name, Vec<std::pair<RcPtr<Inst>,BigRational>> &&operands );
    static RcPtr<Inst>  from_operands( const Str &name, Vec<RcPtr<Inst>> operands );
 
    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual void        display      ( Str &res, int prio = 0 ) const override;
    virtual RcPtr<Inst> clone        ( const Vec<RcPtr<Inst>> &new_children ) const override;
    virtual int         type         () const override { return type_Mul; }
};

}