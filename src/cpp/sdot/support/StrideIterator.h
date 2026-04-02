#pragma once

#include "common_types.h"
#include <iterator>

namespace sdot {

///
template<class T>
struct StrideIterator {
    using            iterator_category = std::random_access_iterator_tag;
    using            difference_type   = SI;
    using            value_type        = T;
    using            reference         = T&;
    using            pointer           = T*;
    using            RawPtr            = std::conditional_t<std::is_const_v<T>,const std::byte*,std::byte*>;

    auto             operator!=        ( const StrideIterator &that ) const { return index != that.index; }
    StrideIterator&  operator++        () { ptr += stride; ++index; return *this; }
    StrideIterator   operator++        ( int ) { auto tmp = *this; ++*this; return tmp; }
    StrideIterator&  operator--        () { ptr -= stride; --index; return *this; }
    reference        operator*         () const { return *reinterpret_cast<T*>( ptr ); }
    SI               operator-         ( const StrideIterator &that ) const { return index - that.index; }

    RawPtr           ptr;
    SI               stride;
    PI               index;
};


} // namespace sdot
