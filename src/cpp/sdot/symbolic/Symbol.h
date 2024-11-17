#pragma once

#include "Inst.h"

namespace sdot {

/** */
class Symbol : public Inst {
public:
    static RcPtr<Inst> from_name( const Str &name );

    virtual void       display  ( Displayer &ds ) const override;
    virtual int        type      () const override { return type_Symbol; }
      
    Str                name;
};

}