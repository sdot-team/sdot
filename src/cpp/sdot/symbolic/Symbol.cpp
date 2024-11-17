#include "Symbol.h"

namespace sdot {

std::map<Str,RcPtr<Inst>> symbol_map;

RcPtr<Inst> Symbol::from_name( const Str &name ) {
    auto iter = symbol_map.find( name );
    if ( iter == symbol_map.end() ) {
        auto *v = new Symbol;
        v->name = name;

        iter = symbol_map.insert( iter, { name, v } );
    }
    return iter->second;
}

void Symbol::display( Displayer &ds ) const {
    ds << name;
}

}
