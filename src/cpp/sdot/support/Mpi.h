#pragma once

// #include "MyPrint.h"
#include <tl/support/containers/Vec.h>
#include <tl/support/TODO.h>
#include "MpiDataInfo.h"
#include "MpiContent.h"
#include <functional>

namespace sdot {

/**
  Allows to use mpi semantic even if mpi is not loaded.

  Use #include "Mpi.h" and mpi.init( argc, argv ) to use MPI
*/
class Mpi {
public:
    virtual int           rank                   () const = 0;
    virtual int           size                   () const = 0;

    // variants that compute and "send" the result only to a given machine (tgt_rank)
    T_T T                 reduction_to           ( PI tgt_rank, const T &value, const std::function<void( T &a, const T &b )> &func, MpiDataInfo data_info = {} );
    T_T auto              gather_to              ( PI tgt_rank, const T &value, auto &&func, MpiDataInfo data_info = {} ); ///< return func( [ value_for_each_process ] ), called only on rank == tgt_rank
    T_T T                 sum_to                 ( PI tgt_rank, const T &value, MpiDataInfo data_info = {} );

    // variants that scatter the results to all machines
    T_T T                 reduction              ( const T &value, const std::function<void( T &a, const T &b )> &func, MpiDataInfo data_info = {} );
    T_T auto              gather                 ( const T &value, auto &&func, MpiDataInfo data_info = {} ); ///< return func( [ value_for_each_process ] )
    T_T T                 sum                    ( const T &value, MpiDataInfo data_info = {} );

    // send
    T_T void              scatter_from           ( PI src_rank, T &value, MpiDataInfo data_info = {} ); ///< `value` is copied from rank `src_rank`

protected:
    using                 B                      = PI8;

    virtual void          _scatter               ( PI src_rank, Span<B> value ) = 0; ///< send value from src_rank to other machines
    virtual void          _gather                ( PI tgt_rank, Span<B> outputs, CstSpan<B> input ) = 0; ///< in tgt_rank, get concatenation of each input in outputs
};

extern Mpi *mpi;

// ------------------------------- IMPL -------------------------------
T_T T Mpi::reduction_to( PI tgt_rank, const T &value, const std::function<void( T &a, const T &b )> &func, MpiDataInfo data_info ) {
    return gather_to( tgt_rank, value, [&]( CstSpan<T> values ) {
        T res = values[ 0 ];
        for( PI r = 1; r < values.size(); ++r )
            f( res, values[ r ] );
        return res;
    }, data_info );
}

T_T auto Mpi::gather_to( PI tgt_rank, const T &value, auto &&func, MpiDataInfo data_info ) {
    using TR = DECAYED_TYPE_OF( func( CstSpan<T>( &value, 1 ) ) );
    if ( size() == 1 )
        return func( CstSpan<T>( &value, 1 ) );

    // homogeneous size
    using MC = MpiContent<DECAYED_TYPE_OF( value )>;
    if constexpr ( MC::dynamic_mpi_size ) {
        // serialize `value`
        return MpiContent<DECAYED_TYPE_OF( value )>::as_mpi( { &value, 1 }, [&]( CstSpan<B> value ) {
            // get all the content in tgt_rank
            Vec<B> room( FromSize(), value.size() * this->size() );
            _gather( tgt_rank, room, value );

            // call func only in tgt_rank
            if ( rank() != tgt_rank )
                return TR{};

            // call func with a CstSpan of T for each rank
            return MpiContent<DECAYED_TYPE_OF( value )>::as_cpp( room, func );
        } );
    } else {
        // heterogeneous size => we have first to gather the sizes
        // data_info.assume_homogeneous_mpi_data_size
        TODO;
    }
}

T_T T Mpi::sum_to( PI tgt_rank, const T &value, MpiDataInfo data_info ) {
    return reduction_to( tgt_rank, value, []( T &a, const T &b ) { a += b; }, data_info );
}


T_T T Mpi::reduction( const T &value, const std::function<void( T &a, const T &b )> &func, MpiDataInfo data_info ) {
    return scatter_from( 0, reduction_to( 0, value, func ), data_info);
}

T_T auto Mpi::gather( const T &value, auto &&func, MpiDataInfo data_info ) {
    return scatter_from( 0, gather_to( 0, value, FORWARD( func ), data_info ) );
}

T_T T Mpi::sum( const T &value, MpiDataInfo data_info ) {
    return scatter_from( 0, sum_to( 0, value, data_info ) );
}

T_T void Mpi::scatter_from( PI src_rank, T &value, MpiDataInfo data_info ) {
    TODO;
}

// template<class OS,class... Args> void ___my_mpi_print( OS &os, const char *str, const Args &...args ) {
//     // get local msg
//     std::ostringstream ss;
//     __my_print( ss, str, ", ", args... );
//     std::string msg = ss.str();

//     // send data (if rank != 0) or display it
//     if ( mpi->rank() ) {
//         mpi->send( (const std::int8_t *)msg.data(), msg.size(), 0, mpi->rank() );
//     } else {
//         // message for rank 0
//         if ( mpi->size() )
//             os << "rank 0: ";
//         os << msg;

//         // other messages
//         for( int rank = 1; rank < mpi->size(); ++rank ) {
//             std::string str( mpi->probe_size( rank, rank ), ' ' );
//             mpi->recv( (std::int8_t *)str.data(), str.size(), rank, rank );
//             os << "rank " << rank << ": " << str;
//         }
//     }

//     // we don't want other displays to interfere
//     mpi->barrier();
// }

// template<class OS,class... Args> void ___my_mpi_print_0( OS &os, const char *str, const Args &...args ) {
//     // get local msg
//     std::ostringstream ss;
//     __my_print( ss, str, ", ", args... );

//     if ( mpi->rank() == 0 )
//         os << ss.str();
// }

// #define PNMPI( ... ) ___my_mpi_print( std::cout, #__VA_ARGS__ " ->\n" , __VA_ARGS__ )
// #define PMPI( ... ) ___my_mpi_print( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )
// #define PMPI_0( ... ) ___my_mpi_print_0( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )

} // namespace sdot
