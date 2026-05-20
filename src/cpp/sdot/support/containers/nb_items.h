#pragma once

#include "../common_types.h"

namespace sdot {

HD PI nb_items( auto &&list ) requires ( requires { list.nb_items(); } || requires { list.size(); } ) {
    if constexpr ( requires { list.nb_items(); } )
        return list.nb_items();
    else
        return list.size();
}

} // namespace sdot

