#pragma once

#include "AxisTuple.h"
#include "Arch.h"

namespace sdot {

/// Batch dispatch object.
/// Batch<Arch, 0>: no batch axes — delegates to arch.run_single, func receives an empty AxisTuple.
/// Batch<Arch, 1>: one batch axis of size `size0` — loops and dispatches each index.
template<class _Arch, int _n_axes = 0, class _TI = PI>
class Batch;

template<class _Arch, class _TI>
class Batch<_Arch, 0, _TI> {
public:
    using TI   = _TI;
    using Arch = _Arch;

    Batch( Arch arch ) : arch( arch ) {}

    void run( auto &&func ) {
        arch.run_single( [func] HD () mutable { func( AxisTuple<TI,Arch,0>( Values() ) ); } );
    }

    Arch arch;
};

template<class _Arch, class _TI>
class Batch<_Arch, 1, _TI> {
public:
    using TI   = _TI;
    using Arch = _Arch;

    Batch( Arch arch, TI size0 ) : arch( arch ), size0( size0 ) {}

    void run( auto &&func ) {
        for ( TI i = 0; i < size0; ++i )
            arch.run_single( [func, i] HD () mutable { func( AxisTuple<TI,Arch,1>( Values(), i ) ); } );
    }

    Arch arch;
    TI   size0;
};

} // namespace sdot
