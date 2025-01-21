#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Add : public Inst {
public:
    static RcPtr<Inst>  from_operands( const BigRational &ca, const RcPtr<Inst> &a, const BigRational &cb, const RcPtr<Inst> &b );
    static RcPtr<Inst>  from_operands( Vec<std::pair<BigRational,RcPtr<Inst>>> &&operands );

    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info    () const override;
    virtual void        display      ( Str &res, int prio = 0 ) const override;
    virtual RcPtr<Inst> clone        ( const Vec<RcPtr<Inst>> &new_children ) const override;
    virtual int         type         () const override;

    Vec<BigRational>    coefficients;
};

}