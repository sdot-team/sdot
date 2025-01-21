#pragma once

#include <sdot/support/BigRational.h>
#include "Inst.h"

namespace sdot {

/** */
class FuncWithCoefficients : public Inst {
public:
    static RcPtr<Inst>  from_operands( const Str &name, Vec<std::pair<RcPtr<Inst>,BigRational>> &&operands );
    static RcPtr<Inst>  from_operands( const Str &name, Vec<RcPtr<Inst>> operands );
 
    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
       
    Vec<BigRational>    coefficients; ///<
};

}