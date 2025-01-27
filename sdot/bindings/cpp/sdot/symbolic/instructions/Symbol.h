#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Symbol : public Inst {
public:
    static RcPtr<Inst>  from_name   ( const Str &name );

    virtual int         compare_same( const Inst &that ) const override;
    virtual void        ct_rt_split ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const override;
    virtual Str         base_info   () const override;
    virtual RcPtr<Inst> clone       ( Vec<RcPtr<Inst>> &&new_children ) const override;
    virtual int         type        () const override { return type_Symbol; }
     
    Str                 name;
};

}