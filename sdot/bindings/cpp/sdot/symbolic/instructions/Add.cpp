#include "Value.h"
#include "Add.h"

namespace sdot {

RcPtr<Inst> Add::from_operands( Vec<std::pair<BigRational,RcPtr<Inst>>> &&operands ) {
    // split operands
    for( auto &operand : operands )
        operand = operand.second->mul_pair( operand.first );

    // 
    // std::sort( operands.begin(), operands.end(), []( const std::pair<RcPtr<Inst>,BigRational> &a, const std::pair<RcPtr<Inst>,BigRational> &b ) {
    //     if ( int c = a.first->compare( *b.first ) )
    //         return c < 0;
    //     return a.second < b.second;
    // } );
    // sort operands

    // all known ?
    Vec<BigRational> values;
    for( const auto &operand : operands )
        if ( Opt<BigRational> v = operand.second->constant_value() )
            values << *v;
        else
            break;
    
    if ( values.size() == operands.size() ) {
        BigRational res = 0;
        for( PI i = 0; i < operands.size(); ++i )
            res += operands[ i ].first * values[ i ];
        return Value::from_value( res );
    }

    // else, 
    auto *res = new Add;
    for( const auto &operand : operands ) {
        res->coefficients << operand.first;
        res->add_child( operand.second );
    }
    return res;
}

RcPtr<Inst> Add::from_operands( const BigRational &ca, const RcPtr<Inst> &a, const BigRational &cb, const RcPtr<Inst> &b ) {
    return from_operands( Vec<std::pair<BigRational,RcPtr<Inst>>>{ { ca, a }, { cb, b } } );
}

void Add::display( Str &res, int prio ) const {
    // std::string res = name;
    // res += '(';
    
    // for( PI i = 0; i < children.size(); ++i ) {
    //     res += i ? ", " : " ";
        
    //     if ( name == "add" ) {
    //         if ( const auto &ch = children[ i ] ) {
    //             if ( coefficients[ i ] != 1 ) {
    //                 if ( coefficients[ i ] == -1 )
    //                     res += "- ";
    //                 else
    //                     res += to_string( coefficients[ i ] ) + " * ";
    //             }
    //             res += to_string( *ch );
    //         } else
    //             res += to_string( coefficients[ i ] );
    //         continue;
    //     }

    //     if ( name == "mul" ) {
    //         if ( const auto &ch = children[ i ] ) {
    //             res += to_string( *ch );

    //             if ( coefficients[ i ] != 1 ) {
    //                 res += " ^ " + to_string( coefficients[ i ] );
    //             }
    //         } else
    //             res += to_string( coefficients[ i ] );
    //         continue;
    //     }

    //     res += to_string( *children[ i ] );
    // }

    // res += " )";
    // ds << res;
    TODO;
}

RcPtr<Inst> Add::clone( const Vec<RcPtr<Inst>> &new_children ) const {
    TODO;
}

Str Add::base_info() const {
    return "Add";
}

int Add::type() const {
    return type_Add;
}

}
