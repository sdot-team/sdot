#include <tl/support/string/to_string.h>
#include <tl/support/P.h>
#include <algorithm>
#include "Value.h"
#include "Mul.h"

namespace sdot {

bool Mul::compare( const Mul &a, const Mul &b ) {
    if ( int c = ::compare( a.children.size(), b.children.size() ) )
        return c;
    if ( int c = ::compare( a.additional_coeff, b.additional_coeff ) )
        return c;
    for( PI i = 0; i < a.children.size(); ++i ) 
        if ( int c = ::compare( *a.children[ i ], *b.children[ i ] ) )
            return c;
    for( PI i = 0; i < a.coefficients.size(); ++i ) 
        if ( int c = ::compare( a.coefficients[ i ], b.coefficients[ i ] ) )
            return c;
    return 0;
}

RcPtr<Inst> Mul::from_operands( const Vec<std::pair<BigRational,RcPtr<Inst>>> &operands, BigRational additional_coeff ) {
    // split operands
    Vec<std::pair<BigRational,RcPtr<Inst>>> sub_operands;
    for( auto &operand : operands ) {
        if ( auto *mul = dynamic_cast<Mul *>( operand.second.get() ) ) {
            for( PI n = 0; n < mul->coefficients.size(); ++n )
                sub_operands << std::pair<BigRational,RcPtr<Inst>>{ operand.first * mul->coefficients[ n ], mul->children[ n ] };
            additional_coeff *= mul->additional_coeff;
        } else
            sub_operands << operand.second->pow_pair( operand.first );
    }

    // sort operands
    std::sort( sub_operands.begin(), sub_operands.end(), []( const std::pair<BigRational,RcPtr<Inst>> &a, const std::pair<BigRational,RcPtr<Inst>> &b ) {
        if ( int c = a.second->compare( *b.second ) )
            return c < 0;
        return a.first < b.first;
    } );

    // all known ?
    Vec<std::pair<BigRational,RcPtr<Inst>>> new_operands;
    for( PI n = 0; n < sub_operands.size(); ++n ) {
        if ( Opt<BigRational> v = sub_operands[ n ].second->constant_value() ) {
            additional_coeff *= pow( *v, sub_operands[ n ].first );
        } else {
            new_operands << sub_operands[ n ];
            while ( n + 1 < sub_operands.size() && new_operands.back().second->always_equal( *sub_operands[ n + 1 ].second ) )
                new_operands.back().first += operands[ ++n ].first;
            // ... ^ 0
            if ( new_operands.back().first == 0 )
                new_operands.pop_back();
        }
    }
    
    // only a constant value
    if ( new_operands.empty() )
        return Value::from_value( additional_coeff );

    // find the same operations in parents
    auto is_same = [&]( Inst *parent ) {
        auto *mul = dynamic_cast<Mul *>( parent );
        if ( ! mul )
            return false;
        if ( mul->children.size() != new_operands.size() )
            return false;
        if ( mul->additional_coeff != additional_coeff )
            return false;
        for( PI n = 0; n < new_operands.size(); ++n )
            if ( new_operands[ n ].first != mul->coefficients[ n ] || new_operands[ n ].second != mul->children[ n ] )
                return false;
        return true;
    };
    for( Inst *parent : new_operands[ 0 ].second->parents )
        if ( is_same( parent ) )
            return parent;

    // else
    auto *res = new Mul;
    res->additional_coeff = additional_coeff;
    for( const auto &operand : new_operands ) {
        res->coefficients << operand.first;
        res->add_child( operand.second );
    }
    return res;
}

RcPtr<Inst> Mul::from_operands( const BigRational &ca, const RcPtr<Inst> &a, const BigRational &cb, const RcPtr<Inst> &b ) {
    return from_operands( Vec<std::pair<BigRational,RcPtr<Inst>>>{ { ca, a }, { cb, b } } );
}

void Mul::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

void Mul::display( Str &res, int prio ) const {
    if ( prio > prio_Mul )
        res += "( ";

    PI o = 0;
    if ( additional_coeff != 0 ) {
        res += additional_coeff.compact_repr();
        o = 1;
    }

    for( PI i = 0; i < children.size(); ++i, ++o ) {
        if ( coefficients[ i ] == 1 ) {
            if ( o )
                res += " * ";
            children[ i ]->display( res, prio_Mul );
        } else if ( coefficients[ i ] == -1 && o ) {
            res += " / ";
            children[ i ]->display( res, prio_Mul + 1 );
        } else {
            if ( o )
                res += " * ";
            children[ i ]->display( res, prio_Pow );
            res += " ** " + coefficients[ i ].compact_repr();
        }
    }

    if ( prio > prio_Mul )
        res += " )";
}

RcPtr<Inst> Mul::clone( const Vec<RcPtr<Inst>> &new_children ) const {
    Vec<std::pair<BigRational,RcPtr<Inst>>> operands;
    for( const RcPtr<Inst> &ch : new_children )
        operands << ch->pow_pair( 1 );
    return from_operands( std::move( operands ), 0 );
}

Str Mul::base_info() const {
    return "Mul";
}

int Mul::type() const {
    return type_Mul;
}

}
