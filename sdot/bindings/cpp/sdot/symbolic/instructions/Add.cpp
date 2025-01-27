#include <tl/support/string/to_string.h>
#include <tl/support/P.h>
#include <algorithm>
#include "Value.h"
#include "Add.h"

namespace sdot {

int Add::compare_same( const Inst &that ) const {
    const auto &b = static_cast<const Add &>( that );
    if ( int c = ::compare( additional_coeff, b.additional_coeff ) )
        return c;
    return ::compare( coefficients, b.coefficients );
}

RcPtr<Inst> Add::from_operands( const Vec<std::pair<BigRational,RcPtr<Inst>>> &operands, BigRational additional_coeff ) {
    // split operands
    Vec<std::pair<BigRational,RcPtr<Inst>>> sub_operands;
    for( auto &operand : operands ) {
        if ( auto *add = dynamic_cast<Add *>( operand.second.get() ) ) {
            for( PI n = 0; n < add->coefficients.size(); ++n )
                sub_operands << std::pair<BigRational,RcPtr<Inst>>{ operand.first * add->coefficients[ n ], add->children[ n ] };
            additional_coeff += add->additional_coeff;
        } else
            sub_operands << operand.second->mul_pair( operand.first );
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
            additional_coeff += sub_operands[ n ].first * *v;
        } else {
            new_operands << sub_operands[ n ];
            while ( n + 1 < sub_operands.size() && new_operands.back().second->always_equal( *sub_operands[ n + 1 ].second ) )
                new_operands.back().first += sub_operands[ ++n ].first;
            // 0 * ...
            if ( new_operands.back().first == 0 )
                new_operands.pop_back();
        }
    }
    
    // only a constant value
    if ( new_operands.empty() )
        return Value::from_value( additional_coeff );

    // find the same operations in parents
    auto is_same = [&]( Inst *parent ) {
        auto *add = dynamic_cast<Add *>( parent );
        if ( ! add )
            return false;
        if ( add->children.size() != new_operands.size() )
            return false;
        if ( add->additional_coeff != additional_coeff )
            return false;
        for( PI n = 0; n < new_operands.size(); ++n )
            if ( new_operands[ n ].first != add->coefficients[ n ] || new_operands[ n ].second != add->children[ n ] )
                return false;
        return true;
    };
    for( Inst *parent : new_operands[ 0 ].second->parents )
        if ( is_same( parent ) )
            return parent;

    // else
    auto *res = new Add;
    res->additional_coeff = additional_coeff;
    for( const auto &operand : new_operands ) {
        res->coefficients << operand.first;
        res->add_child( operand.second );
    }
    return res;
}

RcPtr<Inst> Add::from_operands( const BigRational &ca, const RcPtr<Inst> &a, const BigRational &cb, const RcPtr<Inst> &b ) {
    return from_operands( Vec<std::pair<BigRational,RcPtr<Inst>>>{ { ca, a }, { cb, b } } );
}

void Add::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

void Add::display( Str &res, int prio ) const {
    if ( prio > prio_Add )
        res += "( ";

    for( PI i = 0; i < children.size(); ++i ) {
        // (+/-) coeff * 
        if ( coefficients[ i ] == -1 ) {
            if ( i )
                res += " ";
            res += "- ";

            children[ i ]->display( res, prio_Mul );
        } else if ( coefficients[ i ] == 1 ) {
            if ( i )
                res += " + ";
            children[ i ]->display( res, prio_Add );
        } else {
            if ( coefficients[ i ].is_positive_or_null() ) {
                if ( i )
                    res += " + ";
                res += coefficients[ i ].compact_repr() + " * ";
            } else {
                if ( i )
                    res += " ";
                res += "- " + ( - coefficients[ i ] ).compact_repr() + " * ";
            }
            children[ i ]->display( res, prio_Mul );
        }
    }

    if ( additional_coeff != 0 ) {
        if ( children.size() ) {
            if ( additional_coeff.is_positive_or_null() )
                res += " + " + additional_coeff.compact_repr();
            else
                res += " - " + ( - additional_coeff ).compact_repr();
        } else
            res += additional_coeff.compact_repr();
    }

    if ( prio > prio_Add )
        res += " )";
}

RcPtr<Inst> Add::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    Vec<std::pair<BigRational,RcPtr<Inst>>> operands;
    for( PI i = 0; i < new_children.size(); ++i )
        operands << new_children[ i ]->mul_pair( coefficients[ i ] );
    return from_operands( std::move( operands ), additional_coeff );
}

Str Add::base_info() const {
    return "Add";
}

int Add::type() const {
    return type_Add;
}

}
