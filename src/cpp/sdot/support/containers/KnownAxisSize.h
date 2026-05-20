#pragma once

namespace sdot {

template<class TI,TI offset,class V>
struct WithOffset;

/// Used in TensorShape
///
template<class TI,int num_axis,TI value>
struct KnownAxisSize {
    KnownAxisSize( TI /*value*/ ) {}
    KnownAxisSize() {}
};

template<class TO,TO offset,class TI,int num_axis,TI value>
struct WithOffset<TO,offset,KnownAxisSize<TI,num_axis,value>> {
    using type = KnownAxisSize<TI,num_axis+offset,value>;
};

} // namespace sdot
