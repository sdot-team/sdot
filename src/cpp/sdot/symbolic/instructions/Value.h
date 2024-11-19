#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Value : public Inst {
public:
    static RcPtr<Inst>  from_value    ( const BigRational &value );
 
    virtual auto        constant_value() const -> Opt<BigRational> override;
    virtual bool        always_equal  ( const Inst &that ) const override;
    virtual void        ct_rt_split   ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual void        display       ( Displayer &ds ) const override;
    virtual int         type          () const override { return type_Value; }
    virtual RcPtr<Inst> subs          ( const std::map<Str,RcPtr<Inst>> &map ) const override;
     
    BigRational         value;
};

}
