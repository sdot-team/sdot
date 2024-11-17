#include <tl/support/string/to_string.h>
#include <algorithm>
#include "Func.h"

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
            if ( int c = compare( a.second[ i ], b.second[ i ] ) )
                return c < 0;
        }
        return false;
    }
};

std::map<Key,RcPtr<Inst>,CmpKey> func_map;

RcPtr<Inst> Func::from_operands( const Str &name, Vec<std::pair<RcPtr<Inst>,BigRational>> operands ) {
    std::sort( operands.begin(), operands.end(), []( const std::pair<RcPtr<Inst>,BigRational> &a, const std::pair<RcPtr<Inst>,BigRational> &b ) {
        if ( int c = compare( *a.first, *b.first ) )
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

RcPtr<Inst> Func::from_operands( const Str &name, Vec<RcPtr<Inst>> operands ) {
    Vec<std::pair<RcPtr<Inst>,BigRational>> foperands;
    for( PI i = 0; i < operands.size(); ++i )
        foperands << std::pair<RcPtr<Inst>,BigRational>{ operands[ i ], 1 };
    return from_operands( name, foperands );
}

void Func::display( Displayer &ds ) const {
    std::string res = name;
    res += '(';
    for( PI i = 0; i < children.size(); ++i ) {
        res += i ? ", " : " ";
        if ( coefficients[ i ] != 1 ) {
            res += to_string( coefficients[ i ] ) + " * ";
        }
        res += to_string( *children[ i ] );
    }
    res += " )";
    ds << res;
}

}
