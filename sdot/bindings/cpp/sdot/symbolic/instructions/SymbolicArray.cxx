#pragma once

#include <tl/support/type_info/type_name.h>
#include <tl/support/string/to_string.h>
#include <tl/support/P.h>

#include "SymbolicArray.h"
#include "../ExprData.h"
#include "../Expr.h"
#include "Value.h"
#include "sdot/support/BigRational.h"

namespace sdot {

#define DTP template<class TF,int nb_dims>
#define UTP SymbolicArray<TF,nb_dims>

DTP void UTP::display( Displayer &ds ) const {
    DS_OBJECT( SymbolicArray, extents, values, children );
}

DTP void UTP::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    // PI num_var = rt_data_num( data_map, this, [&]() {
    //     return new ExprVal<const SymbolicArray *>{ this };
    // } );

    // // Inst type
    // cw.write_positive_int( type_SymbolicArray, nb_types );

    // // type info
    // cw.write_string      ( type_name( CtType<TF>() ) );
    // cw.write_positive_int( nb_dims );

    // // number
    // cw.write_positive_int( num_var );
    TODO;
}


DTP RcPtr<Inst> UTP::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    Vec<Expr> indices;
    for( const auto &ch : children )
        indices << ch->subs( map );

    Vec<Opt<BigRational>> kv_indices;
    for( const Expr &ind : indices ) {
        if ( Opt<BigRational> kvi = ind.constant_value() )
            kv_indices << std::move( kvi );
        else
            break;
    }

    // everything is known
    if ( kv_indices.size() == indices.size() ) {
        PI o = 0, m = 1;
        for( PI d = nb_dims; d--; ) {
            PI i( *kv_indices[ d ] );
            if ( i >= extents[ d ] )
                return Value::from_value( 0 );
            o += m * i;
            m *= extents[ d ];
        }

        return Value::from_value( values[ o ] );
    }

    // else, make a new SymbolicArray
    auto *res = new SymbolicArray<TF,nb_dims>;
    res->extents = extents;
    res->values = values;
    for( const Expr &ind : indices )
        res->add_child( ind.inst );
    return res;
}

}
