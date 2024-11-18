#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Value : public Inst {
public:
    static RcPtr<Inst> from_value ( const BigRational &value );

    virtual Str        ct_rt_split( Vec<std::pair<const Inst *,ExprData>> &data_map ) const override;
    virtual void       display    ( Displayer &ds ) const override;
    virtual int        type       () const override { return type_Value; }
      
    BigRational        value;
};

}