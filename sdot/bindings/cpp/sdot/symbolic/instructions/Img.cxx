#pragma once

#include <tl/support/type_info/type_name.h>
#include <tl/support/string/to_string.h>
#include <tl/support/containers/Opt.h>
#include "../ExprData.h"
#include "Value.h"
#include "Img.h"

namespace sdot {

#define DTP template<class TF,int nb_dims,int interpolation_order>
#define UTP Img<TF,nb_dims,interpolation_order>

DTP void UTP::display( Displayer &ds ) const {
    ds << "img";
}

DTP void UTP::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    PI num_var = rt_data_num( data_map, this, [&]() {
        return new ExprVal<const Img *>{ this };
    } );

    // Inst type
    cw.write_positive_int( type_Img, nb_types );

    // type info
    cw.write_string      ( type_name( CtType<TF>() ) );
    cw.write_positive_int( nb_dims );
    cw.write_positive_int( interpolation_order );

    // number
    cw.write_positive_int( num_var );
}


DTP RcPtr<Inst> UTP::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    auto naxis = []( StrView s ) -> Opt<PI> {
        if ( ! s.starts_with( "x_" ) )
            return {};
        PI r = 0;
        for( PI i = 2; i < s.size(); ++i ) {
            if ( s[ i ] < '0' || s[ i ] > '9' )
                return {};
            r = 10 * r + ( s[ i ] - '0' );
        }
        return { r };
    };

    Vec<RcPtr<Inst>,nb_dims> axes_exprs;
    PI nb_kv = 0;
    for( const auto &p : map ) {
        if ( Opt<PI> n = naxis( p.first ) ) {
            if ( *n < nb_dims ) {
                if ( ! axes_exprs[ *n ] )
                    nb_kv += p.second->type() == Inst::type_Value;
                axes_exprs[ *n ] = p.second;
            }
        }
    }

    if ( nb_kv == nb_dims ) {
        Vec<BigRational,nb_dims> axes_values;
        for( PI d = 0; d < nb_dims; ++d )
            axes_values[ d ] = static_cast<const Value *>( axes_exprs[ d ].get() )->value;
        Vec<BigRational,nb_dims> tr_axes_values = trinv( axes_values );

        PI o = 0, m = 1;
        for( PI d = nb_dims; d--; ) {
            SI n( ceil( tr_axes_values[ d ] * extents[ d ] ) );
            if ( n < 0 || n >= extents[ d ] )
                return Value::from_value( 0 );
            o += m * n;
            m *= extents[ d ];
        }

        return Value::from_value( values[ o ] );
    }
    
    if ( nb_kv == 0 )
        return const_cast<Img *>( this );
    
    TODO;
}

}
