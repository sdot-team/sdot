#pragma once

#include "StaticAxisValue.h"
#include "../common_types.h"
#include "../Ct.h"  // IWYU pragma: export

namespace sdot {

/// shape of a tensor: a tuple of axis sizes (the values), nothing more.
///   - if static, it's possible to fix specific values for the axes, using StaticAxisValue
/// For instance AxisValues<int,3,StaticAxisValue<int,2,5>> means
///   - rank 3
///   - axis 2 has compile time value 5
///
/// axes must be sorted. Iteration over the index space is NOT a shape concern: use
/// `shape.all_indices()` to get an IndexRange (for_each_item / nb_items / run_*()).
template<class TI,int ct_rank,class... attributes>
class AxisValues;


namespace DetailsAxisValues  {
    // front-axis value type: TI (runtime) unless ct_tail starts with StaticAxisValue<TI,0,v>
    template<class TI,class...>        struct _FrontAxisValueType                                { using type = TI;       };
    template<class TI,TI v,class... R> struct _FrontAxisValueType<TI,StaticAxisValue<TI,0,v>,R...> { using type = Ct<TI,v>; };

    //
    template<class TI,int ct_rank,class... ct_tail>
    struct _NextAxisValueType {
        using type = AxisValues<TI,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;;
    };

    template<class TI,int ct_rank,TI v,class... ct_tail>
    struct _NextAxisValueType<TI,ct_rank,StaticAxisValue<TI,0,v>,ct_tail...> {
        using type = AxisValues<TI,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;;
    };
} // namespace DetailsAxisValues


/// known ct_rank > 0, runtime front value
///   ct_tail may be empty, or contain higher-index StaticAxisValue entries
template<class _TI,int _ct_rank,class... ct_tail>
class AxisValues {
public:
    SCInt        ct_rank             = _ct_rank;
    using        TI                  = _TI;

    using        FrontValue          = DetailsAxisValues::_FrontAxisValueType<TI,ct_tail...>::type;
    using        NextValues          = DetailsAxisValues::_NextAxisValueType<TI,ct_rank,ct_tail...>::type;

    HD           AxisValues           ( Function, auto &&func, auto index );
    HD           AxisValues           ( Function, auto &&func );
    HD           AxisValues           ( Values, TI front_value, auto... next_values );

    HD void      for_each_item       ( auto &&cb ) const { cb( front_value ); next_values.for_each_item( FORWARD( cb ) ); }
    HD auto      apply_values        ( auto &&cb ) const;
    HD auto      all_indices         () const; ///< the list of all multi-indices, for run_*()
    T_Uu HD auto operator[]          ( Ct<U,u> ) const;
    HD auto      operator[]          ( TI u ) const;
    HD auto      has_value           ( auto &&func ) const;
    HD auto      all_value           ( auto &&func ) const;
    HD void      display             ( auto &os, const char *prefix = nullptr ) const;
    HD auto      size                () const;

    HD auto      front_shape         () const;

    T_Uu HD auto without_index       ( Ct<U,u> ) const;
    HD auto      without_index       ( TI u ) const;

    FrontValue   front_value;
    NextValues   next_values;
};

/// known ct_rank == 0
template<class _TI>
class AxisValues<_TI,0> {
public:
    SCInt        ct_rank             = 0;
    using        TI                  = _TI;

    HD           AxisValues           ( Function, auto &&/*func*/, auto /*index*/ ) {}
    HD           AxisValues           ( Values ) {}

    HD void      for_each_item       ( auto &&/* cb */ ) const {}
    HD auto      apply_values        ( auto &&cb ) const;
    HD auto      all_indices         () const; ///< the list of all multi-indices, for run_*()
    HD auto      operator[]          ( TI /*u*/ ) const -> TI;
    HD auto      all_value           ( auto &&func ) const;
    HD auto      has_value           ( auto &&func ) const;
    HD void      display             ( auto &os, const char *prefix = nullptr ) const;
    HD auto      size                () const;
};

} // namespace sdot

#include "AxisValues.cxx" // IWYU pragma: export
