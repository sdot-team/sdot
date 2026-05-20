#pragma once

namespace sdot {

template<class TI,TI offset,class V>
struct WithOffset;

/// Used in TensorShape
///
template<class TI,int num_axis,TI value>
struct StaticAxisValue {
    StaticAxisValue( TI /*value*/ ) {}
    StaticAxisValue() {}
};

template<class TO,TO offset,class TI,int num_axis,TI value>
struct WithOffset<TO,offset,StaticAxisValue<TI,num_axis,value>> {
    using type = StaticAxisValue<TI,num_axis+offset,value>;
};

} // namespace sdot
