#pragma once

#include "KnownAxisSize.h"
#include "common_types.h"
#include "Ct.h"  // IWYU pragma: export

namespace sdot {

/// shape of a tensor
///   - rank can be dynamic
///   - if static, it's possible to fix specific values for the axes, using ( num_axis, value ) pairs
/// For instance TensorShape<int,Cpu,3,2,5> means
///   - rank 3
///   - axes 2 has compile time value 5
///
/// axes must be sorted
template<class TI,class Arch,int ct_rank,class... attributes>
class AxisTuple;

// front-axis value type: TI (runtime) unless ct_tail starts with KnownAxisSize<TI,0,v>
template<class TI,class...>        struct _FrontAxisValueType                                { using type = TI;       };
template<class TI,TI v,class... R> struct _FrontAxisValueType<TI,KnownAxisSize<TI,0,v>,R...> { using type = Ct<TI,v>; };

//
template<class TI,class Arch,int ct_rank,class... ct_tail>
struct _NextAxisValueType {
    using type = AxisTuple<TI,Arch,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;;
};

template<class TI,class Arch,int ct_rank,TI v,class... ct_tail>
struct _NextAxisValueType<TI,Arch,ct_rank,KnownAxisSize<TI,0,v>,ct_tail...> {
    using type = AxisTuple<TI,Arch,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;;
};

/// unknown ct_rank
template<class _TI,class _Arch>
class AxisTuple<_TI,_Arch,-1> {
public:
    SCInt     ct_rank     = -1;
    using     Arch        = _Arch;
    using     TI          = _TI;

    void      display     ( auto &os ) const;
    // TODO: ...
};

/// known ct_rank > 0, runtime front value
///   ct_tail may be empty, or contain higher-index KnownAxisSize entries
template<class _TI,class _Arch,int _ct_rank,class... ct_tail>
class AxisTuple {
public:
    SCInt        ct_rank             = _ct_rank;
    using        Arch                = _Arch;
    using        TI                  = _TI;

    using        FrontValue          = _FrontAxisValueType<TI,ct_tail...>::type;
    using        NextValues          = _NextAxisValueType<TI,Arch,ct_rank,ct_tail...>::type;

    HD           AxisTuple           ( Function, auto &&func, auto index );
    HD           AxisTuple           ( Function, auto &&func );
    HD           AxisTuple           ( Values, TI front_value, auto... next_values );

    HD void      for_each_index_split( int index, int size, auto &&func ) const;
    HD void      for_each_index      ( auto &&func, auto ...indices_so_far ) const;
    HD auto      apply_values        ( auto &&cb ) const;
    T_Uu HD auto operator[]          ( Ct<U,u> ) const;
    HD auto      operator[]          ( TI u ) const;
    HD auto      has_value           ( auto &&func ) const;
    HD auto      all_value           ( auto &&func ) const;
    HD auto      nb_items            () const;
    HD void      display             ( auto &os, const char *prefix = nullptr ) const;
    HD auto      size                () const;

    HD auto      front_shape         () const;

    T_Uu HD auto without_index       ( Ct<U,u> ) const;
    HD auto      without_index       ( TI u ) const;

    FrontValue   front_value;
    NextValues   next_values;
};

// /// known ct_rank > 0, compile-time front value (KnownAxisSize at offset 0)
// template<class _TI,class _Arch,int _ct_rank,_TI ct_front_value,class... ct_tail>
// class AxisTuple<_TI,_Arch,_ct_rank,KnownAxisSize<_TI,0,ct_front_value>,ct_tail...> {
// public:
//     using        FrontValue          = Ct<_TI,ct_front_value>;
//     SCInt        ct_rank             = _ct_rank;
//     using        Arch                = _Arch;
//     using        TI                  = _TI;

//     using        Next                = AxisTuple<TI,Arch,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;

//     HD           AxisTuple           ( Function, auto &&func, auto index );
//     HD           AxisTuple           ( Function, auto &&func );
//     HD           AxisTuple           ( Values, TI front_value, auto... next_values );

//     HD void      for_each_index_split( int index, int size, auto &&func ) const;
//     HD void      for_each_index      ( auto &&func, auto ...indices_so_far ) const;
//     HD auto      apply_values        ( auto &&cb ) const;
//     T_Uu HD auto operator[]          ( Ct<U,u> ) const;
//     HD auto      operator[]          ( TI u ) const;
//     HD auto      has_value           ( auto &&func ) const;
//     HD auto      all_value           ( auto &&func ) const;
//     HD auto      nb_items            () const;
//     HD void      display             ( auto &os, const char *prefix = nullptr ) const;
//     HD auto      size                () const;

//     HD auto      front_shape         () const;

//     T_Uu HD auto without_index       ( Ct<U,u> ) const;
//     HD auto      without_index       ( TI u ) const;

//     FrontValue   front_value;
//     Next         next_values;
// };

/// known ct_rank == 0
template<class _TI,class _Arch>
class AxisTuple<_TI,_Arch,0> {
public:
    SCInt        ct_rank             = 0;
    using        Arch                = _Arch;
    using        TI                  = _TI;

    HD           AxisTuple           ( Function, auto &&/*func*/, auto /*index*/ ) {}
    HD           AxisTuple           ( Values ) {}

    HD void      for_each_index_split( int index, int size, auto &&func ) const;
    HD void      for_each_index      ( auto &&func, auto ...indices_so_far ) const;
    HD auto      apply_values        ( auto &&cb ) const;
    HD auto      operator[]          ( TI /*u*/ ) const -> TI;
    HD auto      all_value           ( auto &&func ) const;
    HD auto      has_value           ( auto &&func ) const;
    HD auto      nb_items            () const;
    HD void      display             ( auto &os, const char *prefix = nullptr ) const;
    HD auto      size                () const;
};

} // namespace sdot

#include "AxisTuple.cxx" // IWYU pragma: export
