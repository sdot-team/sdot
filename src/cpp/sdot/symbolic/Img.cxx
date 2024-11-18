#pragma once

#include <tl/support/string/to_string.h>
#include <tl/support/P.h>
// #include <algorithm>
#include "Img.h"

namespace sdot {

#define DTP template<class TF,int nb_dims>
#define UTP Img<TF,nb_dims>

DTP void UTP::display( Displayer &ds ) const {
    ds << "img";
}

DTP void UTP::ct_rt_split( CompactReprWriter &cw, Vec<std::pair<const Inst *,ExprData>> &data_map ) const {
    // cw.write_positive_int( type_Func, nb_types );
    // cw << name;

    // cw.write_positive_int( children.size() );
    // for( PI i = 0; i < children.size(); ++i ) {
    //     cw << coefficients[ i ];
    //     children[ i ]->ct_rt_split( cw, data_map );
    // }
    TODO;
}

}
