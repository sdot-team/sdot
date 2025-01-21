#include <tl/support/string/to_string.h>
#include "FuncWithCoefficients.h"

namespace sdot {

using Key = std::pair<Str,Vec<std::pair<RcPtr<Inst>,BigRational>>>;

struct CmpKey {
    bool operator()( const Key &a, const Key &b ) const {
        if ( int c = compare( a.first, b.first ) )
            return c < 0;
        if ( int c = compare( a.second.size(), b.second.size() ) )
            return c < 0;
        for( PI i = 0; i < a.second.size(); ++i ) {
            if ( int c = compare( *a.second[ i ].first, *b.second[ i ].first ) )
                return c < 0;
            if ( int c = BigRational::compare( a.second[ i ].second, b.second[ i ].second ) )
                return c < 0;
        }
        return false;
    }
};

RcPtr<Inst> FuncWithCoefficients::from_operands( const Str &name, Vec<std::pair<RcPtr<Inst>,BigRational>> &&operands ) {
    std::sort( operands.begin(), operands.end(), []( const std::pair<RcPtr<Inst>,BigRational> &a, const std::pair<RcPtr<Inst>,BigRational> &b ) {
        if ( int c = a.first->compare( *b.first ) )
            return c < 0;
        return a.second < b.second;
    } );

    Key key( name, operands );

    auto iter = func_map.find( key );
    if ( iter == func_map.end() ) {
        auto *func = new Func;
        func->name = name;
        for( const auto &o : operands ) {
            func->coefficients << o.second;
            func->add_child( o.first );
        }

        iter = func_map.insert( iter, { key, func } );
    }

    return iter->second;
}

RcPtr<Inst> FuncWithCoefficients::from_operands( const Str &name, Vec<RcPtr<Inst>> operands ) {
    Vec<std::pair<RcPtr<Inst>,BigRational>> foperands;
    for( PI i = 0; i < operands.size(); ++i )
        foperands << std::pair<RcPtr<Inst>,BigRational>{ operands[ i ], 1 };
    return from_operands( name, std::move( foperands ) );
}

void FuncWithCoefficients::display( Displayer &ds ) const {
    std::string res = name;
    res += '(';
    
    for( PI i = 0; i < children.size(); ++i ) {
        res += i ? ", " : " ";
        
        if ( name == "add" ) {
            if ( const auto &ch = children[ i ] ) {
                if ( coefficients[ i ] != 1 ) {
                    if ( coefficients[ i ] == -1 )
                        res += "- ";
                    else
                        res += to_string( coefficients[ i ] ) + " * ";
                }
                res += to_string( *ch );
            } else
                res += to_string( coefficients[ i ] );
            continue;
        }

        if ( name == "mul" ) {
            if ( const auto &ch = children[ i ] ) {
                res += to_string( *ch );

                if ( coefficients[ i ] != 1 ) {
                    res += " ^ " + to_string( coefficients[ i ] );
                }
            } else
                res += to_string( coefficients[ i ] );
            continue;
        }

        res += to_string( *children[ i ] );
    }

    res += " )";
    ds << res;
}

void FuncWithCoefficients::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    cw.write_positive_int( type_Func, nb_types );
    cw << name;

    cw.write_positive_int( children.size() );
    for( PI i = 0; i < children.size(); ++i ) {
        cw << coefficients[ i ];
        children[ i ]->ct_rt_split( cw, data_map );
    }
}

RcPtr<Inst> FuncWithCoefficients::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    TODO;
    return {};
}

}
