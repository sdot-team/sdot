#pragma once

#include "../support/BigRational.h"
#include "Inst.h"

namespace sdot {

/** */
class Expr {
public:
    /**/        Expr     ( const BigRational &value = 0 );
    /**/        Expr     ( const RcPtr<Inst> &inst );
    /**/        Expr     ( const char *expr );
    /**/        Expr     ( StrView expr );
    /**/        Expr     ( int value );
  
    void        display  ( Displayer &ds ) const;

    friend Expr operator+( const Expr &a, const Expr &b );

    RcPtr<Inst> inst;
};

}