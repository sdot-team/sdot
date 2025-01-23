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

DTP Str UTP::base_info() const {
    return "SymbolicArray";
}

DTP RcPtr<Inst> UTP::clone( const Vec<RcPtr<Inst>> &new_children ) const {
    Vec<BigRational> indices;
    for( const RcPtr<Inst> &ch : new_children ) {
        if ( Opt<BigRational> kvi = ch->constant_value() )
            indices << *kvi;
        else
            break;
    }

    // if all indices are known
    if ( indices.size() == children.size() ) {
        PI o = 0, m = 1;
        for( PI d = nb_dims; d--; ) {
            PI i( indices[ d ] );
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
    for( const RcPtr<Inst> &nch : new_children )
        res->add_child( nch );
    return res;
}

}
