#pragma once

#include "support/TaylorScalar.h"
#include "geometry/Bsp.h"
#include "support/P.h"

#include "nanobind_wrappers.h"

namespace sdot {

template<class TF,class AdditionalPtData,int ct_dim>
void get_search_dirs( std::vector<TensorView<TF,1,Cpu>> &search_dirs, Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const auto &primitive ) {
    using Scalar = TaylorScalar<TF,3>;
    std::vector<std::vector<Scalar>> res( search_dirs.size() );
    for( auto &vec : res )
        vec.resize( sorted_potentials.size(), TF( 0 ) );
    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
        auto new_sorted_potentials = [&]( PI index ) {
            Scalar res = sorted_potentials[ index ];
            if ( index == cell.info.local_dirac_index )
                res += Scalar::variable( 1, 0 );
            return res;
        };
        auto mod_cell = bsp.remake_cell( cell, primitive, new_sorted_potentials );
        auto measure = mod_cell.measure();

        if ( measure.c0 <= 0 )
            throw std::runtime_error( "empty cell" );

        auto err = measure - cell.info.dirac_weight;
        res[ 0 ][ cell.info.local_dirac_index ] = err * err;

        //
        // TF der = 0;
        // TF measure = cell.for_each_cut_with_measure( [&]( const auto &cut, const TF cut_measure ) {
        //     if ( cut.info.global_dirac_index == PI( -1 ) )
        //         return;
        //     der += cut_measure / ( 2 * norm_2( cut.info.dirac_position - cell.info.dirac_position ) );
        // } );
        // P( measure, ( cell.info.dirac_weight - measure ) / der );
    } );


    for( PI i = 0; i < sorted_potentials.size(); ++i )
        search_dirs[ 0 ][ i ] = res[ 0 ][ i ].argmin()[ 0 ] - sorted_potentials[ i ];

    // for( PI i = 0; i < sorted_potentials.size(); ++i )
    //     P( search_dirs[ 0 ][ i ] );
}

template<class TF,class AdditionalPtData,int ct_dim>
void get_search_dir( TensorView<TF,1,Cpu> search_dir, Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const auto &primitive ) {
    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
        TF der = 0;
        TF measure = cell.for_each_cut_with_measure( [&]( const auto &cut, const TF cut_measure ) {
            if ( cut.info.global_dirac_index == PI( -1 ) )
                return;
            der += cut_measure / ( 2 * norm_2( cut.info.dirac_position - cell.info.dirac_position ) );
        } );

        if ( measure <= 0 )
            throw std::runtime_error( "empty cell" );

        search_dir[ cell.info.local_dirac_index ] = ( measure - cell.info.dirac_weight ) / der;
    } );
}

template<class TF,class AdditionalPtData,int ct_dim>
void get_system( TensorView<TF,2,Cpu> M, TensorView<TF,1,Cpu> V, Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const std::vector<TensorView<const TF,1,Cpu>> &search_dirs, const auto &primitive ) {
    const PI n = V.size();
    std::vector<TF> loc_ders( n );
    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
        for( TF &d : loc_ders )
            d = 0;
        TF measure = cell.for_each_cut_with_measure( [&]( const auto &cut, const TF cut_measure ) {
            if ( cut.info.global_dirac_index == PI( -1 ) )
                return;
            const TF coeff = cut_measure / ( 2 * norm_2( cut.info.dirac_position - cell.info.dirac_position ) );
            for( PI r = 0; r < n; ++r )
                loc_ders[ r ] += ( search_dirs[ r ][ cell.info.local_dirac_index ] - search_dirs[ r ][ cut.info.local_dirac_index ] ) * coeff;
        } );

        for( PI r = 0; r < n; ++r ) {
            for( PI c = 0; c < n; ++c )
                M( r, c ) += loc_ders[ c ] * loc_ders[ r ];
            V[ r ] += ( measure - cell.info.dirac_weight ) * loc_ders[ r ];
        }
    } );
}

template<class TF,class AdditionalPtData,int ct_dim>
void get_poly_3( TensorView<TF,1,Cpu> poly, Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const std::vector<TensorView<const TF,1,Cpu>> &search_dirs, const auto &primitive ) {
    using Scalar = TaylorScalar<TF,3>;
    std::vector<Scalar> coefficients( search_dirs.size() );
    for( PI i = 0; i < search_dirs.size(); ++i )
        coefficients[ i ] = Scalar::variable( search_dirs.size(), i );

    auto modified_potentials = [&]( PI index ) {
        Scalar res = sorted_potentials[ index ];
        for( PI i = 0; i < search_dirs.size(); ++i )
            res += search_dirs[ i ][ index ] * coefficients[ i ];
        return res;
    };

    Scalar res = 0;
    Cell<TF,1,Cpu> bounds;
    bsp.for_each_cell( primitive, modified_potentials, [&]( auto &cell ) {
        auto measure = cell.measure();

        if ( search_dirs.size() == 1 )
            measure.update_bounds_to_stay_positive( bounds );

        auto err = measure - cell.info.dirac_weight;
        res += err * err;
    } );

    P( bounds );

    PI i = 0;
    poly[ i++ ] = res.c0;
    for( TF v : res.c1 )
        poly[ i++ ] = v;
    for( TF v : res.c2 )
        poly[ i++ ] = v;
    for( TF v : res.c3 )
        poly[ i++ ] = v;
}

template<class TF,class AdditionalPtData,int ct_dim>
void get_poly_4( TensorView<TF,1,Cpu> poly, Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const std::vector<TensorView<const TF,1,Cpu>> &search_dirs, const auto &primitive ) {
    using Scalar = TaylorScalar<TF,4>;
    std::vector<Scalar> coefficients( search_dirs.size() );
    for( PI i = 0; i < search_dirs.size(); ++i )
        coefficients[ i ] = Scalar::variable( search_dirs.size(), i );

    auto modified_potentials = [&]( PI index ) {
        Scalar res = sorted_potentials[ index ];
        for( PI i = 0; i < search_dirs.size(); ++i )
            res += search_dirs[ i ][ index ] * coefficients[ i ];
        return res;
    };

    Scalar res = 0;
    bsp.for_each_cell( primitive, modified_potentials, [&]( auto &cell ) {
        auto err = cell.measure() - cell.info.dirac_weight;
        res += err * err;
    } );

    PI i = 0;
    poly[ i++ ] = res.c0;
    for( TF v : res.c1 )
        poly[ i++ ] = v;
    for( TF v : res.c2 )
        poly[ i++ ] = v;
    for( TF v : res.c3 )
        poly[ i++ ] = v;
    for( TF v : res.c4 )
        poly[ i++ ] = v;
}

template<class TF,class AdditionalPtData,int ct_dim>
void for_each_cell( Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, const std::function<void( typename Bsp<AdditionalPtData,TF,ct_dim,Cpu>::Cell &cell )> &function, TensorView<const TF,1,Cpu> sorted_potentials, const auto &primitive ) {
    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
        function( cell );
    } );
}

template<class TF,class AdditionalPtData,int ct_dim>
void write_vtk( std::string filename, Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const auto &primitive ) {
    VtkOutput vo;
    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
        cell.display_vtk( vo );
    } );
    vo.save( filename );
}

template<class TF,class AdditionalPtData,int ct_dim>
auto ot_system( Bsp<AdditionalPtData,TF,ct_dim,Cpu> &bsp, TensorView<const TF,1,Cpu> sorted_potentials, const auto &primitive ) {
    std::vector<PI> m_rows, m_cols;
    std::vector<TF> m_vals, v_vals;
    m_rows.reserve( sorted_potentials.size() );
    m_cols.reserve( sorted_potentials.size() );
    m_vals.reserve( sorted_potentials.size() );
    v_vals.reserve( sorted_potentials.size() );
    TF E = 0;

    bsp.for_each_cell( primitive, sorted_potentials, [&]( auto &cell ) {
        TF sum_der = 0;
        TF measure = 0;
        primitive.for_each_piece( cell, [&]( auto &piece, const auto &...piece_data ) {
            // facets
            piece.for_each_facet( [&]( const auto &facet ) {
                if ( facet.info.global_dirac_index == PI( -1 ) )
                    return;
                const TF der = primitive.facet_integral( facet, piece_data... ) / ( 2 * norm_2( facet.info.dirac_position - cell.info.dirac_position ) );
                m_cols.push_back( facet.info.local_dirac_index );
                m_rows.push_back( cell.info.local_dirac_index );
                m_vals.push_back( - der );
                sum_der += der;
            } );

            // volume
            measure += primitive.integral( piece, piece_data... );
        } );

        if ( measure <= 0 )
            throw std::runtime_error( "empty cell" );

        m_rows.push_back( cell.info.local_dirac_index );
        m_cols.push_back( cell.info.local_dirac_index );
        m_vals.push_back( sum_der );

        TF err = cell.info.dirac_weight - measure;
        v_vals.push_back( err );
        E += err * err;
    } );

    return std::tuple(
        to_ndarray_1d( std::move( m_rows ) ),
        to_ndarray_1d( std::move( m_cols ) ),
        to_ndarray_1d( std::move( m_vals ) ),
        to_ndarray_1d( std::move( v_vals ) ),
        E
    );
}

} // namespace sdot
