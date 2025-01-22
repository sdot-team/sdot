#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Value : public Inst {
public:
    static RcPtr<Inst>  from_value    ( const BigRational &value );
    static bool         compare       ( const Value &a, const Value &b );

    virtual auto        constant_value() const -> Opt<BigRational> override;
    virtual void        ct_rt_split   ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info     () const override;
    virtual int         type          () const override { return type_Value; }
     
    BigRational         value;
};

}
