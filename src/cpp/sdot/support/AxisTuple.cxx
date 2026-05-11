#pragma once

#include "AxisTuple.h"

#include <utility>

namespace sdot {

// ---- generic with attributes -------------------------------------------------------
#define UTP template<class _TI,class _Arch,int _ct_rank,class... ct_tail>
#define DTP AxisTuple<_TI,_Arch,_ct_rank,ct_tail...>

UTP auto DTP::without_axis( TI u ) const {
    using Res = AxisTuple<TI,Arch,ctd_sub(ct_rank,1)>;
    return apply_values( [&]( auto ...vals ) {
        TI arr[ct_rank] = { TI(vals)... };
        return [&]<std::size_t... Js>( std::index_sequence<Js...> ) {
            return Res( Values(), arr[ Js + ( TI( Js ) >= u ? 1 : 0 ) ]... );
        }( std::make_index_sequence<ct_rank-1>{} );
    } );
}

#undef UTP
#undef DTP

// ---- with KnownAxisSize at front ---------------------------------------------------
#define UTP template<class _TI,class _Arch,int _ct_rank,_TI ct_front_value,class... ct_tail>
#define DTP AxisTuple<_TI,_Arch,_ct_rank,KnownAxisSize<_TI,0,ct_front_value>,ct_tail...>

UTP auto DTP::without_axis( TI u ) const {
    using Res = AxisTuple<TI,Arch,ctd_sub(ct_rank,1)>;
    return apply_values( [&]( auto ...vals ) {
        TI arr[ct_rank] = { TI(vals)... };
        return [&]<std::size_t... Js>( std::index_sequence<Js...> ) {
            return Res( Values(), arr[ Js + ( TI( Js ) >= u ? 1 : 0 ) ]... );
        }( std::make_index_sequence<ct_rank-1>{} );
    } );
}

#undef UTP
#undef DTP

// ---- no attributes -----------------------------------------------------------------
#define UTP template<class _TI,class _Arch,int _ct_rank>
#define DTP AxisTuple<_TI,_Arch,_ct_rank>

UTP auto DTP::without_axis( TI u ) const {
    using Res = AxisTuple<TI,Arch,ctd_sub(ct_rank,1)>;
    return apply_values( [&]( auto ...vals ) {
        TI arr[ct_rank] = { TI(vals)... };
        return [&]<std::size_t... Js>( std::index_sequence<Js...> ) {
            return Res( Values(), arr[ Js + ( TI( Js ) >= u ? 1 : 0 ) ]... );
        }( std::make_index_sequence<ct_rank-1>{} );
    } );
}

#undef UTP
#undef DTP

} // namespace sdot
