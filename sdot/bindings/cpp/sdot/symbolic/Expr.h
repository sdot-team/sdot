#pragma once

#include "../support/BigRational.h"
#include "instructions/Inst.h"

namespace sdot {

/** */
class Expr {
public:
    explicit    Expr          ( const BigRational &value );
    explicit    Expr          ( const RcPtr<Inst> &inst );
    explicit    Expr          ( const char *expr );
    explicit    Expr          ( StrView expr );
    explicit    Expr          ( int value );
    explicit    Expr          ();

    auto        constant_value() const -> Opt<BigRational>;
    bool        always_equal  ( const Expr &that ) const;
    void        display       ( Displayer &ds ) const;
    Expr        subs          ( const std::map<RcPtr<Inst>,RcPtr<Inst>> &map ) const;
     
    friend Expr operator+     ( const Expr &a, const Expr &b );
    friend Expr operator-     ( const Expr &a, const Expr &b );
    friend Expr operator*     ( const Expr &a, const Expr &b );
    friend Expr operator/     ( const Expr &a, const Expr &b );
    friend Expr pow           ( const Expr &a, const Expr &b );

    RcPtr<Inst> inst;
};

}