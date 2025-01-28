#include "Symbol.h"
#include <string>
#include "axis.h"

namespace sdot {

RcPtr<Inst> axis( int n ) {
    static Vec<RcPtr<Inst>> res;
    while ( n >= res.size() )
        res << Symbol::from_name( "x_" + std::to_string( res.size() ) );
    return res[ n ];
}

}
