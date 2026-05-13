#pragma once

#include "KnownAxisSize.h"
#include "CtdInt.h"
#include "Arch.h" // IWYU pragma: export
#include "Ct.h" // IWYU pragma: export

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

//
template<class TI,class Arch,int ct_rank_0,class... attributes_0,int ct_rank_1,class... attributes_1>
auto concat( const AxisTuple<TI,Arch,ct_rank_0,attributes_0...> &t0, const AxisTuple<TI,Arch,ct_rank_1,attributes_1...> &t1 ) {
    using TR = AxisTuple<TI,Arch,ctd_add(ct_rank_0,ct_rank_1),attributes_0...,typename WithOffset<int,ct_rank_0,attributes_1>::type...>;
    return t0.apply_values( [&]( auto ...v0 ) {
        return t1.apply_values( [&]( auto ...v1 ) {
            return TR( Values(), v0..., v1... );
        } );
    } );
}

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

/// known ct_rank, next num known axis is > 0
template<class _TI,class _Arch,int _ct_rank,class... ct_tail>
class AxisTuple {
public:
    SCInt        ct_rank       = _ct_rank;
    using        Arch          = _Arch;
    using        TI            = _TI;

    using        Next          = AxisTuple<TI,Arch,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;

    HD           AxisTuple     ( Function, auto &&func, auto index ) : front_value( func( index ) ), next_values( Function(), FORWARD( func ), index + Ct<int,1>() ) {}
    HD           AxisTuple     ( Function, auto &&func ) : AxisTuple( Function(), func, Ct<int,0>() ) {}
    HD           AxisTuple     ( Values, TI front_value, auto... next_values ) : front_value( front_value ), next_values( Values(), next_values... ) {}

    HD void      for_each_index( auto &&func, auto ...indices_so_far ) const { for( TI i = 0; i < front_value; ++i ) next_values.for_each_index( func, indices_so_far..., i ); }
    HD auto      apply_values  ( auto &&cb ) const { return next_values.apply_values( [&]( auto ...nxt ) { return cb( front_value, nxt... ); } ); }
    T_Uu HD auto operator[]    ( Ct<U,u> ) const { if constexpr ( u == 0 ) return front_value; else return next_values[ Ct<U,u-1>() ]; }
    HD auto      operator[]    ( TI u ) const { if ( u == 0 ) return front_value; return next_values[ u - 1 ]; }
    HD auto      all_value     ( auto &&func ) const { auto res = func( front_value ); if ( ! res ) return res; return next_values.all_value( func ); }
    HD auto      has_value     ( auto &&func ) const { auto res = func( front_value ); if ( res ) return res; return next_values.has_value( func ); }
    HD void      display       ( auto &os, const char *prefix = nullptr ) const { if ( prefix == nullptr ) prefix = "[ "; next_values.display( os << prefix << front_value, ", " ); }
    HD int       size          () const { return ct_rank; }

    HD auto      front_shape   () const { return AxisTuple<TI,Arch,1>( Values(), front_value ); }

    T_Uu HD auto without_axis  ( Ct<U,u> ) const { if constexpr ( u == 0 ) return next_values; else return concat( front_shape(), next_values.without_axis( Ct<U,u-1>() ) ); }
    HD auto      without_axis  ( TI u ) const;

    TI           front_value;
    Next         next_values;
};

/// known ct_rank, next num known axis is 0
template<class _TI,class _Arch,int _ct_rank,_TI ct_front_value,class... ct_tail>
class AxisTuple<_TI,_Arch,_ct_rank,KnownAxisSize<_TI,0,ct_front_value>,ct_tail...> {
public:
    SCInt        ct_rank       = _ct_rank;
    using        Arch          = _Arch;
    using        TI            = _TI;

    using        Next          = AxisTuple<TI,Arch,ct_rank-1,typename WithOffset<int,-1,ct_tail>::type...>;

    HD           AxisTuple     ( Function, auto &&func, auto index ) : next_values( Function(), FORWARD( func ), index + Ct<int,1>() ) { ASSERT_EQ( func( index ), ct_front_value ); }
    HD           AxisTuple     ( Function, auto &&func ) : AxisTuple( Function(), func, Ct<int,0>() ) {}
    HD           AxisTuple     ( Values, TI front_value, auto... next_values ) : next_values( Values(), next_values... ) { ASSERT_EQ( front_value, ct_front_value ); }

    HD void      for_each_index( auto &&func, auto ...indices_so_far ) const { for( TI i = 0; i < ct_front_value; ++i ) next_values.for_each_index( func, indices_so_far..., i ); }
    HD auto      apply_values  ( auto &&cb ) const { return next_values.apply_values( [&]( auto ...nxt ) { return cb( ct_front_value, nxt... ); } ); }
    T_Uu HD auto operator[]    ( Ct<U,u> ) const { if constexpr ( u == 0 ) return Ct<TI,ct_front_value>(); else return next_values[ Ct<U,u-1>() ]; }
    HD auto      operator[]    ( TI u ) const { if ( u == 0 ) return ct_front_value; return next_values[ u - 1 ]; }
    HD auto      all_value     ( auto &&func ) const { auto res = func( Ct<int,ct_front_value>() ); if ( ! res ) return res; return next_values.all_value( func ); }
    HD auto      has_value     ( auto &&func ) const { auto res = func( Ct<int,ct_front_value>() ); if ( res ) return res; return next_values.has_value( func ); }
    HD void      display       ( auto &os, const char *prefix = nullptr ) const { if ( prefix == nullptr ) prefix = "[ "; next_values.display( os << prefix << ct_front_value, ", " ); }
    HD int       size          () const { return ct_rank; }

    HD auto      front_shape   () const { return AxisTuple<TI,Arch,1,KnownAxisSize<TI,0,ct_front_value>>( Values(), ct_front_value ); }

    T_Uu HD auto without_axis  ( Ct<U,u> ) const { if constexpr ( u == 0 ) return next_values; else return concat( front_shape(), next_values.without_axis( Ct<U,u-1>() ) ); }
    HD auto      without_axis  ( TI u ) const;

    Next         next_values;
};

/// known ct_rank > 0, no next known axis
template<class _TI,class _Arch,int _ct_rank>
class AxisTuple<_TI,_Arch,_ct_rank> {
public:
    SCInt        ct_rank       = _ct_rank;
    using        Arch          = _Arch;
    using        TI            = _TI;

    using        Next          = AxisTuple<TI,Arch,ct_rank-1>;

    HD           AxisTuple     ( Function, auto &&func, auto index ) : front_value( func( index ) ), next_values( Function(), FORWARD( func ), index + Ct<int,1>() ) {}
    HD           AxisTuple     ( Function, auto &&func ) : AxisTuple( Function(), func, Ct<int,0>() ) {}
    HD           AxisTuple     ( Values, TI front_value, auto... next_values ) : front_value( front_value ), next_values( Values(), next_values... ) {}

    HD void      for_each_index( auto &&func, auto ...indices_so_far ) const { for( TI i = 0; i < front_value; ++i ) next_values.for_each_index( func, indices_so_far..., i ); }
    HD auto      apply_values  ( auto &&cb ) const { return next_values.apply_values( [&]( auto ...nxt ) { return cb( front_value, nxt... ); } ); }
    T_Uu HD auto operator[]    ( Ct<U,u> ) const { if constexpr ( u == 0 ) return front_value; else return next_values[ Ct<U,u-1>() ]; }
    HD auto      operator[]    ( TI u ) const { return ( &front_value )[ u ]; }
    HD auto      all_value     ( auto &&func ) const { auto res = func( front_value ); if ( ! res ) return res; return next_values.all_value( func ); }
    HD auto      has_value     ( auto &&func ) const { auto res = func( front_value ); if ( res ) return res; return next_values.has_value( func ); }
    HD void      display       ( auto &os, const char *prefix = nullptr ) const { if ( prefix == nullptr ) prefix = "[ "; next_values.display( os << prefix << front_value, ", " ); }
    HD int       size          () const { return ct_rank; }

    HD auto      front_shape   () const { return AxisTuple<TI,Arch,1>( Values(), front_value ); }

    T_Uu HD auto without_axis  ( Ct<U,u> ) const { if constexpr ( u == 0 ) return next_values; else return concat( front_shape(), next_values.without_axis( Ct<U,u-1>() ) ); }
    HD auto      without_axis  ( TI u ) const;

    TI           front_value;
    Next         next_values;
};

/// known ct_rank == 0
template<class _TI,class _Arch>
class AxisTuple<_TI,_Arch,0> {
public:
    SCInt        ct_rank       = 0;
    using        Arch          = _Arch;
    using        TI            = _TI;

    HD           AxisTuple     ( Function, auto &&/*func*/, auto /*index*/ ) {}
    HD           AxisTuple     ( Values ) {}

    HD void      for_each_index( auto &&func, auto ...indices_so_far ) const { func( indices_so_far... ); }
    HD auto      apply_values  ( auto &&cb ) const { return cb(); }
    HD auto      all_value     ( auto &&/*func*/ ) const { return Ct<bool,true>(); }
    HD auto      has_value     ( auto &&/*func*/ ) const { return Ct<bool,false>(); }
    HD auto      operator[]    ( TI /*u*/ ) const -> TI { ASSERT( false ); return 0; }
    HD void      display       ( auto &os, const char *prefix = nullptr ) const { if ( prefix == nullptr ) os << "[]"; else os << " ]"; }
    HD int       size          () const { return 0; }
};

} // namespace sdot

#include "AxisTuple.cxx" // IWYU pragma: export
