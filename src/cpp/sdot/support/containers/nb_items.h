#pragma once

#include "../common_types.h"

namespace sdot {

namespace detail {
    template<class T,class=void> struct has_nb_items_method : std::false_type {};
    T_T struct has_nb_items_method<T,void_t<decltype(std::declval<T>().nb_items())>> : std::true_type {};
}

HD T_T auto nb_items( T &&list ) -> std::enable_if_t<detail::has_nb_items_method<T>::value || detail::has_size_method<T>::value,PI> {
    if constexpr ( detail::has_nb_items_method<T>::value )
        return list.nb_items();
    else
        return list.size();
}

} // namespace sdot

