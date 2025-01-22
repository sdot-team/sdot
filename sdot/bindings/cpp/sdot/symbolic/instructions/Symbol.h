#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Symbol : public Inst {
public:
    static RcPtr<Inst>  from_name  ( const Str &name );
    static bool         compare    ( const Symbol &a, const Symbol &b );

    virtual void        ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info  () const override;
    virtual RcPtr<Inst> clone      ( const Vec<RcPtr<Inst>> &new_children ) const override;
    virtual int         type       () const override { return type_Symbol; }
     
    Str                 name;
};

}