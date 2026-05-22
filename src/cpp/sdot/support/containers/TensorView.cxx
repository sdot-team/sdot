#pragma once

// #include "StrideIterator.h"
// #include "StrideIterator.h"
// #include "CrossArchCopy.h"
#include "../hardware/current_execution_context.h"
#include "../hardware/Run.h"
// #include "contiguous_strides.h"
// #include "IndexRange.h"
#include "CartesianProduct.h"
#include "TensorView.h"
#include "Range.h"

// #include "../ASSERT.h"
// #include "TODO.h"

// #include <algorithm>
// #include <cstring>

#define UTP template<class TF,class Shape,class Strides,class MemorySpace>
#define DTP TensorView<TF,Shape,Strides,MemorySpace>

namespace sdot {

UTP HD DTP DTP::make_invalid( Shape shape, Strides strides, MemorySpace memory_space ) {
    return TensorView( _sentinel().template as<TF>(), shape, strides, memory_space );
}

UTP HD DTP::TensorView( TF *data, Shape shape, Strides strides, MemorySpace memory_space ) : _memory_space( memory_space ), _raw_ptr( reinterpret_cast<RawByte *>( data ) ), _strides( strides ), _shape( shape ) {
}

UTP HD auto DTP::operator()( const auto &index, auto ...rem ) const {
    if constexpr ( requires { DECAYED_TYPE_OF( index.size() )::value; } ) {
        if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
            return operator()( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
        else
            return operator()( rem... );
    } else
        return row( index )( rem... );
}

UTP HD auto DTP::squeeze( auto axis, PI index ) const {
    auto new_strides = _strides.without_index( axis );
    auto new_shape = _shape.without_index( axis );

    using NewStrides = DECAYED_TYPE_OF( new_strides );
    using NewShape = DECAYED_TYPE_OF( new_shape );

    auto ptr = ( _raw_ptr + _strides[ axis ] * index ).template as<TF>();

    return TensorView<TF,NewShape,NewStrides,MemorySpace>( ptr, new_shape, new_strides, _memory_space );
}

UTP HD auto DTP::row( PI index ) const {
    return squeeze( Ct<int,0>(), index );
}

UTP HD auto DTP::data() const {
    return Ptr<TF,MemorySpace>( _raw_ptr.template as<TF>(), _memory_space );
}

UTP HD TF DTP::value() const {
    static_assert( ct_rank == 0 );
    auto ec = current_execution_context();
    if constexpr ( DECAYED_TYPE_OF( accessible_from( ec, _memory_space ) )::value )
        return *data();
    else {
        // #ifdef __CUDA_ARCH__
        //         // unreachable: the static dispatch keeps non-device-accessible data off the device
        //         // (and there is no cudaMemcpy on the device to pull it in).
        //         return TF{};
        // #else
        // host: pull a single element in (D2H copy) and return it
        TF res;
        auto ms = native_memory_space( ec );
        copy( Ptr( &res, ms ), data(), 1 );
        return res;
        // #endif
    }
}

UTP TF& DTP::ref() const {
    static_assert( ct_rank == 0 );
    return *data();
}

UTP void DTP::display( std::ostream &os ) const {
    if constexpr ( ! DECAYED_TYPE_OF( accessible_from( ExecutionSpace_Cpu{}, _memory_space ) )::value ) {
        make_accessible( ExecutionSpace_Cpu{}, [&]( auto &&tensor ) {
            tensor.display( os );
        } );
    } else if constexpr ( ct_rank == 0 ) {
        os << value();
    } else if constexpr ( ct_rank == 1 ) {
        for( TI i = 0; i < shape( 0_c ); ++i )
            sdot::display( os << ( i ? ", " : "" ), operator[]( i ) );
    } else {
        for( std::size_t i = 0; i < shape( 0_c ); ++i )
            sdot::display( os << "\n  ", operator[]( i ) );
    }
}

UTP HD auto DTP::nb_items() const {
    return product( _shape );
}

UTP void DTP::copy_elements_from( const auto &that ) const {
    if ( _strides == that._strides && is_contiguous() ) {
        copy( data(), that.data(), nb_items() );
    } else  {
        TODO; // run( ... ) ?
        // auto ec = current_execution_context();
        // make_accessible( ec, [&]( auto &&a ) {
        //     make_accessible( ec, [&]( auto &&b ) {

        //     } );
        // } );
        // auto av = accessible_from( , memory_space() );
        // auto bv = accessible_from( current_execution_context(), that.memory_space() );
        // if constexpr ( DECAYED_TYPE_OF( av && bv )::value ) {
        //     for_each_index( [&]( auto &&index ) {
        //         operator[]( index ).ref() == that[ index ].value();
        //     } );
        // } else {
        // }
    }
}

UTP HD auto DTP::is_contiguous() const {
    return _strides == contiguous_strides<TF>( _shape );
}

UTP HD void DTP::for_each_index( auto &&func ) const {
    cartesian_product( map( _shape, range<PI> ) ).for_each_item( FORWARD( func ) );
}

// namespace details::TensorView {
//     // Namespace-scope functors for arch-aware element-wise ops.
//     // Lambda bodies inside HD/GD template methods from .cxx files cause issues with
//     // some nvcc versions when the lambda references class-level template params (TF).
//     // Using concrete struct operator() avoids the problem.
//     template<class DstTV, class SrcTV, class BI>
//     struct TensorCopyFunctor {
//         DstTV dst;
//         SrcTV src;
//         GD void operator()( BI bi ) const {
//             dst( bi ).item() = src( bi ).item();
//         }
//     };

//     template<class DstTV, class ValT, class BI>
//     struct TensorFillFunctor {
//         DstTV dst;
//         ValT value;
//         GD void operator()( BI bi ) const {
//             dst( bi ).item() = value;
//         }
//     };
// } // namespace details::TensorView


// UTP void DTP::make_accessible( auto execution_space, auto &&func ) const {
//     if constexpr ( DECAYED_TYPE_OF( accessible_from( execution_space, _memory_space ) )::value ) {
//         // data already reachable from this execution space → no transfer
//         func( *this );
//     } else {
//         // materialize a contiguous copy in the execution space's native memory space, bring
//         // the data in (transfer driven by execution_space's stream), run, then release.
//         // NB: assumes a contiguous source; strided cross-space transfer is a TODO.
//         auto dst_ms = native_memory_space( execution_space );
//         using DstMS = DECAYED_TYPE_OF( dst_ms );
//         dst_ms.template with_reservation<TF>( nb_items(), [&]( Ptr<TF,DstMS> buf ) {
//             auto strides = contiguous_strides<TF>( _shape );
//             TensorView<TF,Shape,DECAYED_TYPE_OF( strides ),DstMS> dst( buf.raw, _shape, strides, dst_ms );
//             copy( execution_space, buf, Ptr<TF,MemorySpace>( data(), _memory_space ), nb_items() );
//             func( dst );
//             // TODO in/out: copy dst → *this back after func, depending on the argument category
//         } );
//     }
// }

// UTP CPU_ONLY void DTP::get_data_from( const auto &that ) {
//     // contiguous -> copy works for all the cases
//     if ( is_contiguous() && that.is_contiguous() ) {
//         copy( data(), that.data(), nb_items() );
//         return;
//     }

//     // same memory space -> for each on indices
//     run_sequential( _shape.all_indices(), details::TensorView::TensorCopyFunctor( *this, that ) );

//     // else
//     TODO;
// }

// UTP void DTP::fill_with( TF value ) {
//     run_parallel( _shape.all_indices(), details::TensorView::TensorFillFunctor{ *this, value } );
// }

// UTP void DTP::with_same_shape( const auto &arch, auto &&func ) const {
//     arch.template with_reservation<TF>( nb_items(), [&]( auto buf ) {
//         auto new_strides = contiguous_strides<TF>( _shape );
//         using NewStrides = DECAYED_TYPE_OF( new_strides );
//         using BufMS      = typename DECAYED_TYPE_OF( buf )::MemorySpace;
//         TensorView<TF,Shape,NewStrides,BufMS> res( buf.raw, _shape, new_strides, buf.memory_space );
//         func( res );
//     } );
// }

// UTP HD void DTP::spill_to( TensorView &that ) {
//     that.get_data_from( *this );
//     _raw_ptr = that._raw_ptr;
// }


// // operator[] and item() are single-context inline accessors in TensorView.h (SDOT_DATA_ACCESSOR)

// UTP HD Strides DTP::strides() const {
//     return _strides;
// }

// UTP HD SI DTP::stride( auto d ) const {
//     return _strides[ d ];
// }

// UTP HD PI DTP::nb_items() const {
//     PI res = 1;
//     for( PI d = 0; d < rank(); ++d )
//         res *= shape( d );
//     return res;
// }

// UTP HD auto DTP::size() const {
//     static_assert( ct_rank == 1, "..." );
//     return shape( Ct<int,0>() );
// }

// UTP HD bool DTP::surely_null() const {
//     if ( is_invalid() )
//         return true;

//     /* Version using lambdas and Ct<> (causes nvcc to crash in some cases)
//     // empty tensor (any dimension == 0)
//     if ( _shape.has_value( []( auto size ) -> bool { return size < Ct<int,1>(); } ) )
//         return true;
//     // all strides zero (rank > 0) → surely-null by construction: all elements alias data()[0] == 0
//     if ( rank() > 0 && ! _strides.has_value( []( auto size ) -> bool { return size != Ct<int,0>(); } ) )
//         return true;
//     // single scalar: check value
//     if ( ! _shape.has_value( []( auto size ) -> bool { return size > Ct<int,1>(); } ) )
//         return *data() == 0;
//     */

//     // empty tensor (any dimension == 0)
//     for ( PI i = 0; i < ct_rank; ++i )
//         if ( _shape[ i ] == 0 )
//             return true;

//     // all strides zero (rank > 0) → surely-null if *data()
//     bool all_strides_zero = true;
//     for ( PI i = 0; i < ct_rank; ++i ) {
//         if ( _strides[ i ] && _shape[ i ] > 1 ) {
//             all_strides_zero = false;
//             break;
//         }
//     }
//     if ( all_strides_zero )
//         return *data() == 0;

//     return false;
// }

// UTP HD bool DTP::is_invalid() const {
//     return _raw_ptr == sentinel();
// }

// UTP HD bool DTP::is_valid() const {
//     return _raw_ptr != sentinel();
// }

// UTP HD auto DTP::empty() const {
//     return _shape.has_value( []( auto size ) { return size == Ct<int,0>(); } );
// }

// UTP HD auto DTP::rank() const {
//     return Ct<int,ct_rank>();
// }


// UTP HD auto DTP::begin() const {
//     if constexpr ( ct_rank == 1 || ct_rank == 0 ) {
//         return data();
//     } else {
//         TODO;
//     }
// }

// UTP HD auto DTP::end() const {
//     if constexpr ( ct_rank == 0 ) {
//         return data() + 1;
//     } else if constexpr ( ct_rank == 1 ) {
//         return data() + size();
//     } else {
//         TODO;
//     }
// }


// UTP HD auto DTP::unsqueeze( auto axis ) const {
//     // Append a dimension of size 1.
//     // The stride for the new axis is sizeof(T): matches contiguous layout when the source is contiguous.
//     // For a size-1 axis the stride value is irrelevant for correctness, but we set it consistently.
//     // constexpr int new_ct_rank = ct_rank >= 0 ? ct_rank + 1 : -1;
//     // return TensorView<T,new_ct_rank,Arch>( data(), _shape.with_pushed_value( PI( 1 ) ), _strides.with_pushed_value( SI( sizeof( T ) ) ) );
//     TODO;
// }


// UTP HD void DTP::for_each_index( auto &&func ) const {
//     IndexRange<Shape>{ _shape }.for_each_item( FORWARD( func ) );
// }

// // #ifdef __CUDACC__
// // // Functor at namespace scope so CUDA/cudafe can properly mangle the type in Thrust typedefs
// // // (local structs with __device__ members cause "template argument N is invalid" in cudafe)
// // template<class T, int ct_rank>
// // struct TensorViewIndexer {
// //     __device__ const T &operator()( sdot::PI i ) const {
// //         sdot::SI off = 0;
// //         for ( int d = ct_rank - 1; d >= 0; --d ) {
// //             off += sdot::SI( i % ext[ d ] ) * str[ d ];
// //             i /= ext[ d ];
// //         }
// //         return *reinterpret_cast<const T *>( src + off );
// //     }

// //     const std::byte *src;
// //     sdot::PI ext[ ct_rank ];
// //     sdot::SI str[ ct_rank ];
// // };

#undef UTP
#undef DTP

} // namespace sdot
