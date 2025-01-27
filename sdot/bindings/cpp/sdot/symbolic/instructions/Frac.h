#pragma once

#include "Inst.h"

namespace sdot {

/**
 */
class Frac : public Inst {
public:
    static RcPtr<Inst>  from_operands( const RcPtr<Inst> &a );

    virtual void        ct_rt_split  ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info    () const override;
    virtual RcPtr<Inst> clone        ( Vec<RcPtr<Inst>> &&new_children ) const override;
    virtual int         type         () const override;
};

}