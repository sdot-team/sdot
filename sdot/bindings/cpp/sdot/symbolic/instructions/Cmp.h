#pragma once

#include "Inst.h"

namespace sdot {

/**
    compare to 0 (-> unary operation)
 */
class Cmp : public Inst {
public:
    enum class          CmpType      { Equal, Inf, InfEq, Sup, SupEq, Neq };

    static RcPtr<Inst>  from_operands( CmpType cmp_type, const RcPtr<Inst> &a );
    static bool         compare      ( const Cmp &a, const Cmp &b );

    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info    () const override;
    virtual void        display      ( Str &res, int prio = 0 ) const override;
    virtual RcPtr<Inst> clone        ( const Vec<RcPtr<Inst>> &new_children ) const override;
    virtual int         type         () const override;

    CmpType             cmp_type;    ///<
};

}