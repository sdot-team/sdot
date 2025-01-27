#pragma once

#include "Inst.h"

namespace sdot {

/**
 */
class Mul : public Inst {
public:
    static RcPtr<Inst>  from_operands( const BigRational &ca, const RcPtr<Inst> &a, const BigRational &cb, const RcPtr<Inst> &b );
    static RcPtr<Inst>  from_operands( const Vec<std::pair<BigRational,RcPtr<Inst>>> &operands, BigRational additional_coeff = 1 );

    virtual int         compare_same ( const Inst &that ) const override;
    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info    () const override;
    virtual void        display      ( Str &res, int prio = 0 ) const override;
    virtual RcPtr<Inst> clone        ( Vec<RcPtr<Inst>> &&new_children ) const override;
    virtual int         type         () const override;

    BigRational         additional_coeff;
    Vec<BigRational>    coefficients; ///< coefficient for each child
};

}