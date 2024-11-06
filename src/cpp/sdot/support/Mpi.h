#pragma once

// #include "MyPrint.h"
#include <tl/support/containers/Vec.h>
#include "MpiContent.h"
#include <functional>

namespace sdot {

/**
  Allows to use mpi semantic even if mpi is not loaded.

  Use #include "Mpi.h" and mpi.init( argc, argv ) to use MPI
*/
class Mpi {
public:
    struct                DataInfo               { bool assume_homogeneous_mpi_data_size = false; };

    bool                  main                   () const { return rank() == 0; }
    virtual int           rank                   () const = 0;
    virtual int           size                   () const = 0;

    // variants that compute and "send" the result only to a given machine (tgt_rank)
    T_T T                 reduction_to           ( PI tgt_rank, const T &value, const std::function<void( T &a, const T &b )> &func, DataInfo data_info = {} );
    T_T auto              gather_to              ( PI tgt_rank, const T &value, auto &&func, DataInfo data_info = {} ); ///< return func( [ value_for_each_process ] ), called only on rank == tgt_rank
    T_T T                 sum_to                 ( PI tgt_rank, const T &value, DataInfo data_info = {} );

    // variants that scatter the results to all machines
    T_T T                 reduction              ( const T &value, const std::function<void( T &a, const T &b )> &func, DataInfo data_info = {} );
    T_T auto              gather                 ( const T &value, auto &&func, DataInfo data_info = {} ); ///< return func( [ value_for_each_process ] )
    T_T T                 sum                    ( const T &value );

    // send
    T_T void              scatter_from           ( PI src_rank, T &value, DataInfo data_info = {} ); ///< `value` is copied from rank `src_rank`

protected:
    using                 B                      = PI8;

    virtual void          _scatter               ( Span<B> value, PI src_rank ) = 0; ///< 
    virtual void          _gather                ( Span<B> output, CstSpan<B> input, PI tgt_rank ) = 0; ///< 
};

extern Mpi *mpi;

// ------------------------------- IMPL -------------------------------
T_T T Mpi::reduction_to( PI tgt_rank, const T &value, const std::function<void( T &a, const T &b )> &func ) {
    return gather_to( tgt_rank, value, [&]( CstSpan<T> values ) {
        T res = values[ 0 ];
        for( PI r = 1; r < values.size(); ++r )
            f( res, values[ r ] );
        return res;
    } );
}

T_T auto Mpi::gather_to( PI tgt_rank, const T &value, auto &&func ) {
    if ( rank() == 1 )
        return func( CstSpan<T>( &value, 1 ) );

    return MpiContent<DECAYED_TYPE_OF( value )>::as_mpi( CstSpan<T>( &value, 1 ), [&]( CstSpan<B> value ) {
        Vec<B> room( FromSize(), value.size() * rank() );
        _gather( room, value );

        using R = DECAYED_TYPE_OF( func( CstSpan<T>( &value, 1 ) ) );
        R res;
        if ( main() ) {
            MpiContent<DECAYED_TYPE_OF( value )>::as_cpp( room, [&]( CstSpan<T> room ) {
                res = func( room );
            } );
        }

        return ;
    } );
}

T_T T Mpi::sum_to( PI tgt_rank, const T &value ) {
    return reduction_to( tgt_rank, value, []( T &a, const T &b ) { a += b; } );
}


T_T T Mpi::reduction( const T &value, const std::function<void( T &a, const T &b )> &func ) {
    return scatter_from( 0, reduction_to( 0, value, func ) );
}

T_T auto Mpi::gather( const T &value, auto &&func ) {
    return scatter_from( 0, gather_to( 0, value, FORWARD( func ) ) );
}

T_T T Mpi::sum( const T &value ) {
    return scatter_from( 0, sum_to( 0, value ) );
}

T_T auto Mpi::scatter_from( PI src_rank, const T &value ) {
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
