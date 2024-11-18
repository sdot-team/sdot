#include "Symbol.h"
#include "sdot/symbolic/Inst.h"

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

void Symbol::ct_rt_split( CompactReprWriter &cw, Vec<std::pair<const Inst *,ExprData>> &data_map ) const {
    cw.write_positive_int( type_Symbol, nb_types );
    cw << name;
}

void Symbol::display( Displayer &ds ) const {
    ds << name;
}

}
