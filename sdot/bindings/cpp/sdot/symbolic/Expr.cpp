#include <tl/support/ERROR.h>
#include <tl/support/P.h>

#include "instructions/Symbol.h"
#include "instructions/Value.h"
#include "instructions/Add.h"
#include "instructions/Mul.h"
#include "instructions/Cmp.h"

#include "Expr.h"

namespace sdot {

enum { sym, ope, num, paro, parc };

struct ParseItem {
    void display( Displayer &ds ) const { DS_OBJECT( type, str ); }

    int type;
    Str str;
};

RcPtr<Inst> parse_rec( Vec<ParseItem> &items ) {
    if ( items.size() == 0 )
        ERROR( "void expression" )

    //
    if ( items.size() == 1 ) {
        const ParseItem &item = items[ 0 ];
        if ( item.type == num )
            return Value::from_value( BigRational( item.str ) );
        if ( item.type == sym )
            return Symbol::from_name( item.str );
        ERROR( "invalid expression" );
    }

    //
    TODO;
}

inline bool is_let( char c ) { return ( c >= 'a' && c <= 'z' ) || ( c >= 'A' && c <= 'Z' ) || c == '_'; }
inline bool is_num( char c ) { return c >= '0' && c <= '9'; }
inline bool is_alp( char c ) { return is_let( c ) || is_num( c ); }
inline bool is_ope( char c ) { return c == '+' || c == '-' || c == '*' || c == '/' || c == '^'; }
inline bool is_spa( char c ) { return c == ' '; }

Expr::Expr( const RcPtr<Inst> &inst ) : inst( inst ) {
}

Expr::Expr( const BigRational &value ) {
    inst = Value::from_value( value );
}

Expr::Expr() {
    inst = Value::from_value( 0 );
}

Expr::Expr( StrView expr ) {
    Vec<ParseItem> items;

    for( PI i = 0; i < expr.size(); ) {
        char c = expr[ i ];
        
        if ( is_spa( c ) ) {
            ++i;
            continue;
        }

        if ( is_let( c ) ) {
            PI o = i;
            while ( ++i < expr.size() && is_alp( expr[ i ] ) );
            items << ParseItem{ sym, Str{ expr.substr( o, i - o ) } };
            continue;
        }

        if ( is_num( c ) ) {
            PI o = i;
            while ( ++i < expr.size() && is_num( expr[ i ] ) );
            items << ParseItem{ num, Str{ expr.substr( o, i - o ) } };
            continue;
        }

        if ( is_ope( c ) ) {
            items << ParseItem{ ope, Str{ expr.substr( i, 1 ) } };
            ++i;
            continue;
        }

        if ( c == '(' ) {
            items << ParseItem{ paro, Str{ expr.substr( i, 1 ) } };
            ++i;
            continue;
        }

        if ( c == ')' ) {
            items << ParseItem{ parc, Str{ expr.substr( i, 1 ) } };
            ++i;
            continue;
        }

        P( c );
        TODO;
    }

    inst = parse_rec( items );
}

Expr::Expr( const char *str ) : Expr( StrView( str ) ) {
}

Expr::Expr( int value ) {
    inst = Value::from_value( value );
}

void Expr::display( Displayer &ds ) const {
    Str res;
    if ( inst )
        inst->display( res );
    else
        res = "NULL";
    ds << res;
}

Expr Expr::subs( const std::map<RcPtr<Inst>,RcPtr<Inst>> &map ) const {
    return Expr{ inst->subs( map ) };
}

bool Expr::always_equal( const Expr &that ) const {
    return inst->always_equal( *that.inst );
}

Opt<BigRational> Expr::constant_value() const {
    return inst->constant_value();
}

Expr operator==( const Expr &a, const Expr &b ) { Expr d = a - b; return Expr{ Cmp::from_operands( Cmp::CmpType::Equal, d.inst ) }; }
Expr operator<=( const Expr &a, const Expr &b ) { Expr d = a - b; return Expr{ Cmp::from_operands( Cmp::CmpType::InfEq, d.inst ) }; }
Expr operator>=( const Expr &a, const Expr &b ) { Expr d = a - b; return Expr{ Cmp::from_operands( Cmp::CmpType::SupEq, d.inst ) }; }
Expr operator!=( const Expr &a, const Expr &b ) { Expr d = a - b; return Expr{ Cmp::from_operands( Cmp::CmpType::Neq, d.inst ) }; }
Expr operator< ( const Expr &a, const Expr &b ) { Expr d = a - b; return Expr{ Cmp::from_operands( Cmp::CmpType::Inf, d.inst ) }; }
Expr operator> ( const Expr &a, const Expr &b ) { Expr d = a - b; return Expr{ Cmp::from_operands( Cmp::CmpType::Sup, d.inst ) }; }

Expr operator+( const Expr &a, const Expr &b ) { return Expr{ Add::from_operands( +1, a.inst, +1, b.inst ) }; }
Expr operator-( const Expr &a, const Expr &b ) { return Expr{ Add::from_operands( +1, a.inst, -1, b.inst ) }; }
Expr operator*( const Expr &a, const Expr &b ) { return Expr{ Mul::from_operands( +1, a.inst, +1, b.inst ) };; }
Expr operator/( const Expr &a, const Expr &b ) { return Expr{ Mul::from_operands( +1, a.inst, -1, b.inst ) };; }
Expr pow( const Expr &a, const Expr &b ) { TODO; }


Expr operator-( const Expr &a ) { return Expr{ Add::from_operands( { { -1, a.inst } } ) }; }

// Expr operator*( const Expr &a, const Expr &b ) { return Expr{ Mul::from_operands( 1, a.inst, +1, b.inst ) }; }
// Expr operator/( const Expr &a, const Expr &b ) { return Expr{ Mul::from_operands( 1, a.inst, -1, b.inst ) }; }
// Expr pow( const Expr &a, const Expr &b ) { return Expr( expr_pow( a.inst, b.inst ) ); }

}
