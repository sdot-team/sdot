#include <tl/support/ERROR.h>
#include <tl/support/P.h>

#include "instructions/Symbol.h"
#include "instructions/Value.h"

#include "Expr.h"

#include "expr_add.h"
#include "expr_mul.h"
#include "expr_pow.h"

namespace sdot {

enum { sym, ope, num, paro, parc };

struct ParseItem {
    void display( Displayer &ds ) const { DS_OJBECT( type, str ); }

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
inline bool is_ope( char c ) { return c == '+' || c == '+' || c == '*' || c == '/' || c == '^'; }
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
    ds << inst;
}

Expr Expr::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    return Expr{ inst->subs( map ) };
}

Expr operator+( const Expr &a, const Expr &b ) { return Expr{ expr_add( 1, a.inst, +1, b.inst ) }; }
Expr operator-( const Expr &a, const Expr &b ) { return Expr{ expr_add( 1, a.inst, -1, b.inst ) }; }
Expr operator*( const Expr &a, const Expr &b ) { return Expr{ expr_mul( 1, a.inst, +1, b.inst ) }; }
Expr operator/( const Expr &a, const Expr &b ) { return Expr{ expr_mul( 1, a.inst, -1, b.inst ) }; }
Expr pow( const Expr &a, const Expr &b ) { return Expr( expr_pow( a.inst, b.inst ) ); }

}
