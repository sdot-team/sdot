#pragma once

#include "support/Matrix.h"
#include "support/index.h"

#include "Cell/CellBoundary.h"
#include "Cell/integral.h"
#include "Cell/Simplex.h"
#include "Cell.h"

#include <numeric>

namespace sdot {

#define UTP PARAMETERS_DECLARATION_OF_Cell
#define DTP Cell<PARAMETER_NAMES_OF_Cell>

UTP void DTP::init_as_aligned_simplex( TI cut_id ) {
    is_fully_closed = cut_id != CellBoundary::INFINITE;
    nb_vertices = dim + 1;
    nb_edges = ( dim + 1 ) * dim / 2;
    nb_cuts = dim + 1;

    // vertex_positions
    for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
        for( PI d = 0; d < dim; ++d )
            vertex_positions( num_vertex, d ) = ( d + 1 == num_vertex );

    // vertex_inds
    if ( dim != 2 )
        for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            for( PI d = 0; d < dim; ++d )
                vertex_indices( num_vertex, d ) = d + ( d >= num_vertex );

    // edge_indices
    if ( dim != 2 ) {
        for ( PI a = 0, o = 0; a < nb_vertices; ++a ) {
            for ( PI b = a + 1; b < nb_vertices; ++b ) {
                if ( a != b ) {
                    edge_indices( o, 0 ) = a;
                    edge_indices( o, 1 ) = b;
                    for( PI d = 0; d < dim - 1; ++d )
                        edge_indices( o, 2 + d ) = d + ( d >= a ) + ( d >= b - 1 );
                    ++o;
                }
            }
        }
    }

    // cut_planes
    if ( dim != 2 ) {
        for( PI num_cut = 0; num_cut < dim; ++num_cut ) {
            for( PI d = 0; d < dim; ++d )
                cut_planes( num_cut, d ) = - ( d == num_cut );
            cut_planes( num_cut, dim ) = 0;
        }
        for( PI d = 0; d < dim + 1; ++d )
            cut_planes( dim, d ) = 1;
    } else {
        cut_planes( 0, 0 ) =  0; cut_planes( 0, 1 ) = -1; cut_planes( 0, 2 ) = 0;
        cut_planes( 1, 0 ) = +1; cut_planes( 1, 1 ) = +1; cut_planes( 1, 2 ) = 1;
        cut_planes( 2, 0 ) = -1; cut_planes( 2, 1 ) =  0; cut_planes( 2, 2 ) = 0;
    }

    // cut_ids
    for( PI num_cut = 0; num_cut < nb_cuts; ++num_cut )
        cut_ids( num_cut ) = cut_id;
}

UTP void DTP::init_as_hypercube( const auto &frame, const auto &cut_id ) {
    is_fully_closed = cut_id != CellBoundary::INFINITE;
    nb_vertices = PI( 1 ) << dim;
    nb_edges = dim * ( PI( 1 ) << ( dim - 1 ) );
    nb_cuts = 2 * dim;

    // shared: F^T[r][c] = axis_c[r], used to compute rows of F^{-1} via solve_ge
    auto FT = Matrix<TF,Arch,ct_dim_value>::with_func( [=] HD ( PI r, PI c ) {
        return TF( frame( 1 + c, r ) );
    } );

    // vertex_positions: origin + sum of selected axes; vertex_indices: cut 2b or 2b+1 per axis
    const PI vertex_ordering_2D[] = { 0, 1, 3, 2 };
    for ( PI k = 0; k < nb_vertices; ++k ) {
        const PI l = ( dim != 2 ? k : vertex_ordering_2D[ k ] );
        for ( PI d = 0; d < dim; ++d ) {
            TF pos = frame( 0, d );
            for ( PI b = 0; b < dim; ++b )
                if ( ( k >> b ) & 1 )
                    pos += frame( 1 + b, d );
            vertex_positions( l, d ) = pos;
        }
    }

    // vertex_indices
    if ( dim >= 2 ) {
        for ( PI k = 0; k < nb_vertices; ++k )
            for ( PI b = 0; b < dim; ++b )
                vertex_indices( k, b ) = 2 * b + ( ( k >> b ) & 1 );
    }

    // edge_indices: edges in direction b, from vertex k (bit b=0) to k|(1<<b)
    if ( dim >= 2 ) {
        for ( PI b = 0, e = 0; b < dim; ++b ) {
            for ( PI k = 0; k < nb_vertices; ++k ) {
                if ( ( k >> b ) & 1 )
                    continue;
                edge_indices( e, 0 ) = k;
                edge_indices( e, 1 ) = k | ( PI( 1 ) << b );
                for ( PI d = 0, col = 2; d < dim; ++d ) {
                    if ( d == b )
                        continue;
                    edge_indices( e, col++ ) = 2 * d + ( ( k >> d ) & 1 );
                }
                ++e;
            }
        }
    }

    // cut planes: row d of F^{-1} via shared FT
    const PI cut_ordering_2D[] = { 3, 1, 0, 2 };
    for ( PI d = 0; d < dim; ++d ) {
        auto e_d = Vector<TF,Arch,ct_dim_value>::with_func( [d] HD ( PI i ) {
            return i == d ? TF( 1 ) : TF( 0 );
        } );
        const auto row = FT.solve_ge( e_d );

        TF row_dot_origin = 0;
        for ( PI c = 0; c < dim; ++c )
            row_dot_origin += row[ c ] * frame( 0, c );

        const PI r0 = ( dim != 2 ? 2 * d + 0 : cut_ordering_2D[ 2 * d + 0 ] );
        for ( PI c = 0; c < dim; ++c )
            cut_planes( r0, c ) = -row[ c ];
        cut_planes( r0, dim ) = -row_dot_origin;
        cut_ids( r0 ) = cut_id;

        const PI r1 = ( dim != 2 ? 2 * d + 1 : cut_ordering_2D[ 2 * d + 1 ] );
        for ( PI c = 0; c < dim; ++c )
            cut_planes( r1, c ) = row[ c ];
        cut_planes( r1, dim ) = row_dot_origin + 1;
        cut_ids( r1 ) = cut_id;
    }
}

UTP void DTP::init_as_hypercube_bwd( const auto &frame, auto &p, const auto &batch_index ) {
    // Fwd: F[r][c] = frame(1+r,c),  A = F^{-1},  row_d = A[d,:] via F^T * row_d = e_d.
    //
    // grad_frame(0,c)   = Σ_l gV(l,c)  +  Σ_d G_dot_d * A[d,c]
    // grad_frame(1+b,c) = Σ_{k: bit b} gV(l(k),c)  -  (A · tGR^T · A)[b,c]
    //
    // where G_dot_d = gC(r1,dim) - gC(r0,dim),
    //       tGR[d,c] = gC(r1,c) - gC(r0,c) + G_dot_d * frame(0,c).

    using Mat = Matrix<TF,Arch,dim>;
    using Vec = Vector<TF,Arch,dim>;

    auto gV = p.input_grad_for_cell_vertex_positions( batch_index );
    auto gC = p.input_grad_for_cell_cut_planes( batch_index );
    auto gF = p.output_grad_for_frame( batch_index );

    const PI vertex_ordering_2D[] = { 0, 1, 3, 2 };
    const PI cut_ordering_2D[]    = { 3, 1, 0, 2 };

    gF.fill_with( TF( 0 ) );

    // vertex positions: gF(0,c) += Σ_l gV(l,c);  gF(1+b,c) += Σ_{k: bit b} gV(l(k),c)
    if ( ! gV.surely_null() ) {
        for ( PI k = 0; k < ( PI( 1 ) << dim ); ++k ) {
            const PI l = ( dim != 2 ? k : vertex_ordering_2D[ k ] );
            for ( PI c = 0; c < dim; ++c ) {
                const TF g = gV( l, c );
                gF( 0, c ) += g;
                for ( PI b = 0; b < dim; ++b )
                    if ( ( k >> b ) & 1 )
                        gF( 1 + b, c ) += g;
            }
        }
    }

    // cut planes: gF(0,c) += G_dot_d * A[d,c];  gF(1+b,c) -= (A tGR^T A)[b,c]
    if ( ! gC.surely_null() ) {
        auto FT  = Mat::with_func( [=] HD ( PI r, PI c ) { return TF( frame( 1 + c, r ) ); } );
        auto A   = Mat( FillWith(), TF( 0 ) );
        auto tGR = Mat( FillWith(), TF( 0 ) );

        for ( PI d = 0; d < dim; ++d ) {
            const PI r0    = ( dim != 2 ? 2 * d + 0 : cut_ordering_2D[ 2 * d + 0 ] );
            const PI r1    = ( dim != 2 ? 2 * d + 1 : cut_ordering_2D[ 2 * d + 1 ] );
            const TF G_dot = gC( r1, dim ) - gC( r0, dim );

            const auto row = FT.solve_ge( Vec::with_func( [d] HD ( PI i ) { return i == d ? TF( 1 ) : TF( 0 ); } ) );
            for ( PI c = 0; c < dim; ++c ) {
                A  ( d, c ) = row[ c ];
                tGR( d, c ) = gC( r1, c ) - gC( r0, c ) + G_dot * frame( 0, c );
                gF ( 0, c ) += G_dot * row[ c ];
            }
        }

        // gF(1+b, c) -= (A tGR^T A)[b,c] = Σ_i (A[b,:] · tGR[i,:]) * A[i,c]
        for ( PI b = 0; b < dim; ++b )
            for ( PI i = 0; i < dim; ++i ) {
                TF s = 0;
                for ( PI c = 0; c < dim; ++c )  s += A( b, c ) * tGR( i, c );
                for ( PI c = 0; c < dim; ++c )  gF( 1 + b, c ) -= s * A( i, c );
            }
    }
}

UTP void DTP::init_as_unbounded() {
    init_as_aligned_simplex( CellBoundary::INFINITE );
}

UTP typename DTP::Pt DTP::vertex_position( PI num_vertex ) const {
    return vertex_positions.row( num_vertex );
}

UTP typename DTP::Ci DTP::vertex_cuts( PI num_vertex ) const {
    if constexpr ( ct_dim == 2 )
        return { Values(), ( num_vertex + nb_vertices - 1 ) % nb_vertices, num_vertex };
    return vertex_indices.row( num_vertex );
}

UTP bool DTP::vertex_inf( PI num_vertex ) const {
    Ci ci = vertex_indices( num_vertex );
    for ( PI d = 0; d < dim; ++d )
        if ( cut_ids( ci[ d ] ) == CellBoundary::INFINITE )
            return true;
    return false;
}

UTP typename DTP::Pt DTP::cut_dir( PI num_cut ) const {
    return std::span( &cut_planes( num_cut, 0 ), dim );
}

UTP TF DTP::cut_dot( PI num_cut ) const {
    return cut_planes( num_cut, dim );
}

UTP TI DTP::cut_id( PI num_cut ) const {
    return cut_ids( num_cut );
}

UTP bool DTP::already_in_simplex( auto &simplex, PI simplex_size, PI next_num_vertex ) {
    for( PI simplex_ind = 0; simplex_ind < simplex_size; ++simplex_ind )
        if ( next_num_vertex == simplex[ simplex_ind ] )
            return true;
    return false;
}

/// Fan triangulation — recursive core.
UTP T_d HD void DTP::for_each_simplex_rec( const Vector<TI,Arch,d> &cut_indices, auto &simplex, PI simplex_size, PI num_vertex, auto &item_map, auto &&func ) {
    // register the new vertex
    simplex[ simplex_size++ ] = num_vertex;

    if constexpr ( d == 0 ) {
        func( simplex );
    } else {
        for( PI ind_to_remove = 0; ind_to_remove < d; ++ind_to_remove ) {
            // first time we see this item -> use vertex `num_vertex`as reference for this item
            auto new_cut_indices = cut_indices.without_index( ind_to_remove );
            auto ic = item_map[ new_cut_indices ];
            if ( ! ic ) {
                ic = num_vertex;
                continue;
            }


            // else, try to make a new simplex
            const PI next_num_vertex = ic;
            if( already_in_simplex( simplex, simplex_size, next_num_vertex ) )
                continue;

            // and continue the recursion
            for_each_simplex_rec( new_cut_indices, simplex, simplex_size, next_num_vertex, item_map, func );
        }
    }
}

UTP HD void DTP::for_each_simplex( RecursiveMapOfUniqueSortedIndices<ct_dim_value-1,TI,Arch> &item_map, auto &&func ) {
    constexpr int ct_simplex = dim + 1;
    if ( nb_vertices == 0 )
        return;

    //
    if ( dim == 2 ) {
        Vector<TI,Arch,ct_simplex> simplex;
        simplex[ 0 ] = 0;
        for( TI num_vertex = 3; num_vertex <= nb_vertices; ++num_vertex ) {
            simplex[ 1 ] = num_vertex - 2;
            simplex[ 2 ] = num_vertex - 1;
            func( simplex );
        }
        return;
    }

    // make a list
    Vector<TI,Arch,ct_simplex> simplex;
    item_map.reserve( nb_vertices );
    for( TI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex ) {
        Vector<TI,Arch,dim> cut_indices( vertex_indices( num_vertex ) );
        for_each_simplex_rec( cut_indices, simplex, 0, num_vertex, item_map, func );
    }
}

UTP void DTP::for_each_facet( auto &&func ) {
    const PI nb_vertices = nb_vertices;

    if ( dim == 2 ) {
        Simplex<ct_dim,dim,TF,Arch> simplex;
        for( TI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex ) {
            simplex.pts[ 0 ] = vertex_position( num_vertex );
            simplex.pts[ 1 ] = vertex_position( ( num_vertex + 1 ) % nb_vertices );
            func( simplex, cut_id( num_vertex ) );
        }
        return;
    }

    TODO;
}

UTP void DTP::for_each_face( auto &&func ) {
    if ( dim == 2 ) {
        std::vector<PI> indices( nb_vertices() );
        std::iota( indices.begin(), indices.end(), 0 );
        func( indices, Vector<TI,Arch,0>( Values() ) );
        return;
    }

    TODO;
    // workspace.nb_links = 0;

    // // on remplit une liste chainée d'edges
    // // pour chaque face, on récupère l'index dans la liste chainée
    // MapOfUniqueSortedIndices<( ct_dim > 0 ? ct_dim - 2 : -1 ),TI,Arch> face_map( workspace.map_items, workspace.nb_map_items, dim - 2, nb_cuts );
    // face_map.reserve_full_capacity();
    // for( PI edge = 0; edge < nb_edges; ++edge ) {
    //     Vector<TI,( ct_dim >= 0 ? ct_dim - 1 : -1 ),Arch> edge_cut_indices = std::span( &cell.edge_indices( edge, 2 ), dim - 1 );
    //     for( PI ind_to_remove = 0; ind_to_remove < dim - 1; ++ind_to_remove ) {
    //         auto face_cut_indices = edge_cut_indices.without_index( ind_to_remove );
    //         auto index_in_links = face_map[ face_cut_indices ];

    //         const TI prev = index_in_links.has_a_value() ? TI( index_in_links ) : -1;
    //         const TI indl = workspace.nb_links.post_increment( 2 );
    //         workspace.links[ indl + 0 ] = prev;
    //         workspace.links[ indl + 1 ] = edge;
    //         index_in_links = indl;
    //     }
    // }

    // face_map.for_each_item( [&]( auto face_cut_indices, TI index_in_links ) {
    //     // reserve some room to store the vertices
    //     PI nb_vertices = 0;
    //     for ( TI idx = index_in_links; idx != TI( -1 ); idx = workspace.links[ idx ] )
    //         ++nb_vertices;
    //     const PI scratch = workspace.nb_links.post_increment( nb_vertices );

    //     const TI e0 = TI( workspace.links[ index_in_links + 1 ] );
    //     const TI vs = cell.edge_indices( e0, 0 ); // start vertex
    //     TI prev_edge = e0;
    //     TI vc = vs; // current vertex

    //     // first vertex
    //     nb_vertices = 0;
    //     workspace.links[ scratch + nb_vertices++ ] = vc;

    //     // extend the vertex chain until it closes; use a raw loop + break to stop early
    //     do {
    //         for ( TI idx = index_in_links; idx != TI( -1 ); idx = workspace.links[ idx ] ) {
    //             const TI e1 = TI( workspace.links[ idx + 1 ] );
    //             if ( e1 == prev_edge )
    //                 continue;

    //             const TI v0 = cell.edge_indices( e1, 0 );
    //             const TI v1 = cell.edge_indices( e1, 1 );
    //             if ( v0 == vc ) {
    //                 prev_edge = e1;
    //                 vc = v1;
    //                 break;
    //             }
    //             if ( v1 == vc ) {
    //                 prev_edge = e1;
    //                 vc = v0;
    //                 break;
    //             }
    //         }

    //         workspace.links[ scratch + nb_vertices++ ] = vc;
    //     } while ( vc != vs );

    //     func( std::span( &workspace.links[ scratch ], nb_vertices - 1 ), face_cut_indices );
    // } );

}

UTP HD TF DTP::measure( RecursiveMapOfUniqueSortedIndices<ct_dim_value-1,TI,Arch> &item_map ) {
    const TI nb_vertices = this->nb_vertices();

    // infinite cell
    if ( ! is_fully_closed() )
        return std::numeric_limits<TF>::max();

    // 2D: shoelace formula
    if ( dim == 2 ) {
        TF sum = 0;
        for ( TI i = 0; i < nb_vertices; ++i ) {
            const TI j = ( i + 1 ) % nb_vertices;
            sum += vertex_positions( i, 0 ) * vertex_positions( j, 1 )
                 - vertex_positions( j, 0 ) * vertex_positions( i, 1 );
        }
        return sum / 2;
    }

    // nD: fan triangulation
    TF sum = 0;
    TF *p_sum = &sum; // captured by value in GD lambda (same device thread → valid pointer)
    for_each_simplex( item_map, [&] GD ( const auto &simplex ) {
        const TI v0 = simplex[ 0 ];
        auto M = Matrix<TF,Arch,dim>::with_func( [&] HD ( TI row, TI col ) {
            return vertex_positions( simplex[ col + 1 ], row ) - vertex_positions( v0, row );
        } );
        *p_sum += std::abs( M.determinant() );
    } );

    return sum / factorial( dim );
}

UTP T_d auto DTP::simplex_from_indices( const Vector<TI,Arch,d> &indices ) const {
    Simplex<ct_dim,d,TF,Arch> res;
    for( PI i = 0; i < d; ++i )
        res.pts[ i ] = vertex_position( indices[ i ] );
    return res;
}

UTP bool DTP::contains( const Pt &p ) const {
    for( PI num_cut = 0; num_cut < nb_cuts; ++num_cut )
        if ( dot( cut_dir( num_cut ), p ) - cut_dot( num_cut ) > 0 )
            return false;
    return true;
}

UTP DTP::Pt DTP::centroid() {
    Pt res( Size(), dim, 0 );
    TF mea = 0;
    for_each_simplex( [&]( const auto &indices ) {
        auto simplex = simplex_from_indices( indices );
        TF m = simplex.measure();
        res += m * simplex.centroid();
        mea += m;
    } );
    return res / mea;
}

UTP void DTP::check_if_fully_closed() {
    for( TI num_cut = 0; num_cut < nb_cuts; ++num_cut )
        if ( cut_ids[ num_cut ] == CellBoundary::INFINITE )
            return;
    is_fully_closed() = true;
}

UTP void DTP::cut( const auto &cut_dir, auto cut_dot, SI cut_id ) {
    // check to grow enough so that all the vertices stay on the same side even if we grow more
    if ( ! is_fully_closed() )
        grow_infinite_cuts( cut_dir, cut_dot );

    //
    TI nb_out = scalar_products( cut_dir, cut_dot );

    if ( nb_out == 0 )
        return;

    if ( nb_out == nb_vertices )
        return clear_cell();

    // 2D shortcut: vertices ordered CCW, cut_planes(k,:) = edge k→(k+1)%nb invariant maintained by cut_2d
    if ( dim == 2 ) {
        cut_2d( cut_dir, cut_dot, cut_id, nb_out );
    } else {
        // store the new cut
        const TI new_cut_index = register_the_new_cut( cut_dir, cut_dot, cut_id );

        // process edges: add new vertices on cut, collect exterior edges into `index_corrections`
        process_edges( new_cut_index );

        // remove exterior vertices, ws.corr = vertex old->new (also updates edge vertex refs)
        remove_unused_vertices( nb_vertices );
        apply_vertex_corr();

        // remove unused cuts, ws.corr = cut old->new; apply to vertex_indices and edge cut indices
        remove_unused_cuts();
        apply_cut_corr();
    }

    // check if closed
    if ( ! is_fully_closed() )
        check_if_fully_closed();
}

UTP PI DTP::scalar_products( auto &sps, const auto &cut_dir, auto cut_dot ) {
    PI nb_out = 0;
    for ( PI v = 0; v < nb_vertices; ++v ) {
        TF sp = vertex_positions( v, 0 ) * cut_dir[ 0 ];
        for ( PI d = 1; d < dim; ++d )
            sp += vertex_positions( v, d ) * cut_dir[ d ];
        sp -= cut_dot;
        sps[ v ] = sp;
        nb_out += ( sp > 0 );
    }
    return nb_out;
}

// generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map
UTP void DTP::swap_and_pop( auto &nb, auto &&move_row ) {
    TODO;
    // const PI nb_initial = PI( nb );
    // ws.reservation = nb_initial;
    // std::iota( ws.corr.data(), ws.corr.data() + nb_initial, 0 );
    // while ( nb_indices_to_remove ) {
    //     const PI dst = indices_to_remove[ --nb_indices_to_remove ];
    //     const PI src = --nb;
    //     ws.corr[ src ] = dst;
    //     if ( dst != src )
    //         move_row( dst, src );
    // }
    // // When a src was itself placed there by a prior step, corr[src]=dst only records
    // // the intermediate hop. Follow each chain to its fixed point (corr[j]==j).
    // // Chains are strictly decreasing (dst<src in every non-trivial step) so no cycles.
    // for ( PI i = 0; i < nb_initial; ++i ) {
    //     PI j = corr[ i ];
    //     while ( corr[ j ] != j )
    //         j = corr[ j ];
    //     corr[ i ] = j;
    // }
}

UTP void DTP::process_edges( PI nc ) {
    // MapOfUniqueSortedIndices<ctd_sub(ct_dim,2),TI,Arch> face_map( map_items, nb_map_items, dim - 2, nc );
    // face_map.reserve( nb_vertices() + nb_edges );

    // // store exterior edge indices in index_corrections,
    // // create the new vertices,
    // // create the new edges
    // nb_index_corrections = 0;
    // for ( PI num_edge = 0; num_edge < nb_edges; ++num_edge ) {
    //     const PI n0 = edge_indices( num_edge, 0 );
    //     const PI n1 = edge_indices( num_edge, 1 );
    //     const TF s0 = sps[ n0 ];
    //     const TF s1 = sps[ n1 ];
    //     const bool e0 = s0 > 0;
    //     const bool e1 = s1 > 0;
    //     if ( e0 == e1 ) {
    //         if ( e0 )
    //             index_corrections[ nb_index_corrections++ ] = num_edge;
    //         continue;
    //     }

    //     // add the new vertex
    //     const TF sc = s0 / ( s1 - s0 );
    //     const PI nn = nb_vertices++;
    //     for ( PI d = 0; d < dim; ++d ) {
    //         const TF p0 = vertex_positions( n0, d );
    //         const TF p1 = vertex_positions( n1, d );
    //         vertex_positions( nn, d ) = p0 - sc * ( p1 - p0 );
    //     }
    //     for ( PI d = 0; d < dim - 1; ++d )
    //         vertex_indices( nn, d ) = edge_indices( num_edge, 2 + d );
    //     vertex_indices( nn, dim - 1 ) = nc;

    //     // update the edge
    //     edge_indices( num_edge, ! e0 ) = nn;

    //     // register the face => vertex correspondance, or create the new edge if already done
    //     for ( PI ind_to_remove = 0; ind_to_remove < dim - 1; ++ind_to_remove ) {
    //         auto face_inds = Vector<PI,Arch,ctd_sub(ct_dim,2)>::with_func( dim - 2, [&]( PI i ) {
    //             return edge_indices( num_edge, 2 + i + ( i >= ind_to_remove ) );
    //         } );

    //         // first time we see the face -> store vertex index
    //         auto face_corr = face_map[ face_inds ];
    //         if ( ! face_corr ) {
    //             face_corr = nn;
    //             continue;
    //         }

    //         // else, create the new edge
    //         const PI ne = nb_index_corrections ? nb_index_corrections-- : nb_edges++;
    //         edge_indices( ne, 0 ) = face_corr;
    //         edge_indices( ne, 1 ) = nn;
    //         for ( PI d = 0; d < dim - 2; ++d )
    //             edge_indices( ne, 2 + d ) = face_inds[ d ];
    //         edge_indices( ne, 2 + dim - 2 ) = nc;
    //     }
    // }

    // // in `index_corrections`, we have the edges that have to be removed (sorted indices)
    // P( index_corrections );
    // exit( 0 );
    // //
}

UTP void DTP::remove_unused_vertices( PI nb_vertices_orig ) {
    TODO;
    // nb_indices_to_remove = 0;
    // for ( PI n = 0; n < nb_vertices_orig; ++n )
    //     if ( sps[ n ] > 0 )
    //         indices_to_remove[ ws.nb_indices_to_remove++ ] = n;

    // // -> update ws.corr
    // swap_and_pop( nb_vertices, [&]( PI dst, PI src ) {
    //     for ( PI d = 0; d < dim; ++d )
    //         vertex_positions( dst, d ) = std::move( vertex_positions( src, d ) );
    //     for ( PI d = 0; d < dim; ++d )
    //         vertex_indices( dst, d ) = std::move( vertex_indices( src, d ) );
    // } );
}

UTP void DTP::apply_vertex_corr() {
    TODO;
    // // update edge vertex references
    // for ( PI e = 0; e < nb_edges; ++e ) {
    //     edge_indices( e, 0 ) = ws.corr[ edge_indices( e, 0 ) ];
    //     edge_indices( e, 1 ) = ws.corr[ edge_indices( e, 1 ) ];
    // }
}

UTP void DTP::remove_unused_cuts() {
    TODO;
    // const PI dim = cell.dim;

    // for( PI i = 0; i < nb_cuts(); ++i )
    // ws.used_flags[ i ] = false;

    // for ( PI v = 0; v < nb_vertices(); ++v )
    //     for ( PI d = 0; d < dim; ++d )
    //         ws.used_flags[ vertex_indices( v, d ) ] = true;

    // ws.nb_indices_to_remove = 0;
    // for ( PI c = 0; c < nb_cuts(); ++c )
    //     if ( ! ws.used_flags[ c ] )
    //         ws.indices_to_remove[ ws.nb_indices_to_remove++ ] = c;

    // swap_and_pop( nb_cuts, [&]( PI dst, PI src ) {
    //     for ( PI d = 0; d <= dim; ++d )
    //         cut_planes( dst, d ) = std::move( cut_planes( src, d ) );
    //     cut_ids( dst ) = std::move( cut_ids( src ) );
    // } );
}

UTP void DTP::apply_cut_corr() {
    TODO;
    // for ( PI v = 0; v < nb_vertices(); ++v )
    //     for ( PI d = 0; d < dim; ++d )
    //         vertex_indices( v, d ) = ws.corr[ vertex_indices( v, d ) ];

    // for ( PI e = 0; e < nb_edges(); ++e )
    //     for ( PI d = 0; d < dim - 1; ++d )
    //         cell.edge_indices( e, 2 + d ) = ws.corr[ cell.edge_indices( e, 2 + d ) ];
}

UTP void DTP::cut_2d( const auto &cut_dir, auto cut_dot, SI cut_id, PI nb_out ) {
    // const SI old_nb_vertices = nb_vertices;

    // // helper to copy vertex_positions and cut_planes/cut_ids together
    // auto copy_vertex_data = [&]( PI dst, PI src ) {
    //     for ( PI d = 0; d < 2; ++d )
    //         vertex_positions( dst, d ) = vertex_positions( src, d );
    // };
    // auto copy_cut_data = [&]( PI dst, PI src ) {
    //     for ( PI d = 0; d < 3; ++d )
    //         cut_planes( dst, d ) = cut_planes( src, d );
    //     cut_ids( dst ) = cut_ids( src );
    // };
    // auto copy_vertex_and_cut_data = [&]( PI dst, PI src ) {
    //     copy_vertex_data( dst, src );
    //     copy_cut_data( dst, src );
    // };

    // auto ext = [&]( PI num_vertex ) {
    //     return sps[ num_vertex ] > 0;
    // };

    // // need to add a vertex
    // if ( nb_out == 1 ) {
    //     const SI n1 = index( sps, []( TF sp ) { return sp > 0; } );
    //     const SI n0 = ( n1 + old_nb_vertices - 1 ) % old_nb_vertices;
    //     const SI n2 = ( n1 + 1 ) % old_nb_vertices;

    //     // compute new vertex positions
    //     const TF sp0 = sps[ n0 ], sp1 = sps[ n1 ], sp2 = sps[ n2 ];
    //     const TF d01 = sp0 / ( sp1 - sp0 );
    //     const TF d12 = sp1 / ( sp2 - sp1 );
    //     TF p01[ 2 ], p12[ 2 ];
    //     for ( PI d = 0; d < 2; ++d ) {
    //         const TF p0 = vertex_positions( n0, d );
    //         const TF p1 = vertex_positions( n1, d );
    //         const TF p2 = vertex_positions( n2, d );
    //         p01[ d ] = p0 - d01 * ( p1 - p0 );
    //         p12[ d ] = p1 - d12 * ( p2 - p1 );
    //     }

    //     // get room for the new point and the new cut
    //     nb_vertices = old_nb_vertices + 1;
    //     nb_edges = old_nb_vertices + 1;
    //     nb_cuts = old_nb_vertices + 1;
    //     for( SI nn = old_nb_vertices; --nn > n1; )
    //         copy_vertex_and_cut_data( nn + 1, nn );
    //     copy_cut_data( n1 + 1, n1 );

    //     // set data for the new point and the new cut
    //     for ( PI d = 0; d < 2; ++d ) {
    //         vertex_positions( n1 + 0, d ) = p01[ d ];
    //         vertex_positions( n1 + 1, d ) = p12[ d ];
    //         cut_planes( n1, d ) = cut_dir[ d ];
    //     }
    //     cut_planes( n1, 2 ) = cut_dot;
    //     cut_ids[ n1 ] = cut_id;
    //     return;
    // }

    // // we stay on the same number of vertices
    // if ( nb_out == 2 ) {
    //     const SI n0 = index( [&]( SI n0 ) { return ! ext( n0 ) && ext( ( n0 + 1 ) % old_nb_vertices ); } );
    //     const SI n1 = ( n0 + 1 ) % old_nb_vertices;
    //     const SI n2 = ( n0 + 2 ) % old_nb_vertices;
    //     const SI n3 = ( n0 + 3 ) % old_nb_vertices;

    //     // compute new vertex positions
    //     const TF sp0 = sps[ n0 ], sp1 = sps[ n1 ], sp2 = sps[ n2 ], sp3 = sps[ n3 ];
    //     const TF d01 = sp0 / ( sp1 - sp0 );
    //     const TF d23 = sp2 / ( sp3 - sp2 );
    //     TF p01[ 2 ], p23[ 2 ];
    //     for ( PI d = 0; d < 2; ++d ) {
    //         const TF p0 = vertex_positions( n0, d );
    //         const TF p1 = vertex_positions( n1, d );
    //         const TF p2 = vertex_positions( n2, d );
    //         const TF p3 = vertex_positions( n3, d );
    //         p01[ d ] = p0 - d01 * ( p1 - p0 );
    //         p23[ d ] = p2 - d23 * ( p3 - p2 );
    //     }

    //     // set data for the new point and the new cut
    //     for ( PI d = 0; d < 2; ++d ) {
    //         vertex_positions( n1, d ) = p01[ d ];
    //         vertex_positions( n2, d ) = p23[ d ];
    //         cut_planes( n1, d ) = cut_dir[ d ];
    //     }
    //     cut_planes( n1, 2 ) = cut_dot;
    //     cut_ids[ n1 ] = cut_id;
    //     return;
    // }

    // SI n0 = index( [&]( SI n0 ) { return ! ext( n0 ) && ext( ( n0 + 1 ) % old_nb_vertices ); } );
    // SI n1 = ( n0 + 1 ) % old_nb_vertices;
    // SI n2 = index( [&]( SI n2 ) { return ext( n2 ) && ! ext( ( n2 + 1 ) % old_nb_vertices ); } );
    // SI n3 = ( n2 + 1 ) % old_nb_vertices;

    // // compute new vertex positions
    // const TF sp0 = sps[ n0 ], sp1 = sps[ n1 ], sp2 = sps[ n2 ], sp3 = sps[ n3 ];
    // const TF d01 = sp0 / ( sp1 - sp0 );
    // const TF d23 = sp2 / ( sp3 - sp2 );
    // TF p01[ 2 ], p23[ 2 ];
    // for ( PI d = 0; d < 2; ++d ) {
    //     const TF p0 = vertex_positions( n0, d );
    //     const TF p1 = vertex_positions( n1, d );
    //     const TF p2 = vertex_positions( n2, d );
    //     const TF p3 = vertex_positions( n3, d );
    //     p01[ d ] = p0 - d01 * ( p1 - p0 );
    //     p23[ d ] = p2 - d23 * ( p3 - p2 );
    // }

    // // remove intermediate points
    // if ( n1 < n2 ) {
    //     const PI nb_to_remove = n2 - ( n1 + 1 ), nb_new_vertices = old_nb_vertices - nb_to_remove;
    //     nb_vertices = nb_new_vertices;
    //     nb_edges = nb_new_vertices;
    //     nb_cuts = nb_new_vertices;

    //     copy_cut_data( n2 - nb_to_remove, n2 );
    //     for( SI nn = n2 + 1; nn < old_nb_vertices; ++nn )
    //         copy_vertex_and_cut_data( nn - nb_to_remove, nn );

    //     // set data for the new point and the new cut
    //     n2 = n1 + 1;
    //     for ( PI d = 0; d < 2; ++d ) {
    //         vertex_positions( n1, d ) = p01[ d ];
    //         vertex_positions( n2, d ) = p23[ d ];
    //         cut_planes( n1, d ) = cut_dir[ d ];
    //     }
    //     cut_planes( n1, 2 ) = cut_dot;
    //     cut_ids[ n1 ] = cut_id;
    // } else {
    //     const PI nb_to_remove = n2 + ( old_nb_vertices - n1 - 1 ), nb_new_vertices = old_nb_vertices - nb_to_remove;
    //     nb_vertices = nb_new_vertices;
    //     nb_edges = nb_new_vertices;
    //     nb_cuts = nb_new_vertices;

    //     if ( n2 ) {
    //         for( SI nn = n2; nn <= n1; ++nn )
    //             copy_vertex_and_cut_data( nn - n2, nn );
    //         n0 -= n2;
    //         n1 -= n2;
    //         n2 = 0;
    //         n3 = 1;
    //     }

    //     // set data for the new point and the new cut
    //     for ( PI d = 0; d < 2; ++d ) {
    //         vertex_positions( n1, d ) = p01[ d ];
    //         vertex_positions( n2, d ) = p23[ d ];
    //         cut_planes( n1, d ) = cut_dir[ d ];
    //     }
    //     cut_planes( n1, 2 ) = cut_dot;
    //     cut_ids[ n1 ] = cut_id;
    // }
}

UTP void DTP::get_data_from( const auto &src_cell ) {
    TODO;
    // vertex_positions.get_data_from( src_vertex_positions, { Values(), src_nb_vertices, dim } );
    // if ( dim != 2 ) {
    //     vertex_indices.get_data_from( src_vertex_indices, { Values(), src_nb_vertices, dim } );
    //     cell.edge_indices.get_data_from( src_cell.edge_indices, { Values(), src_nb_vertices, dim + 1 } );
    // }
    // cut_planes.get_data_from( src_cut_planes, { Values(), src_nb_vertices, dim + 1 } );
    // cut_ids.get_data_from( src_cut_ids, { Values(), src_nb_vertices } );

    // cell.is_fully_closed.get_data_from( src_cell.is_fully_closed );

    // nb_vertices = TI( src_nb_vertices );
    // nb_edges = TI( src_nb_edges() );
    // nb_cuts = TI( src_nb_cuts() );
}

UTP void DTP::clear_cell() {
    is_fully_closed() = 1;
    nb_vertices = 0;
    nb_edges = 0;
    nb_cuts = 0;
}

UTP PI DTP::register_the_new_cut( const auto &cut_dir, auto cut_dot, SI cut_id ) {
    PI res = nb_cuts++;
    for ( PI d = 0; d < dim; ++d )
        cut_planes( res, d ) = cut_dir[ d ];
    cut_planes( res, dim ) = cut_dot;
    cut_ids( res ) = cut_id;
    return res;
}

UTP DTP::Pt DTP::solve_position( PI num_vertex, auto &&add_func ) const {
    Ci ci = vertex_cuts( num_vertex );

    auto M  = Matrix<TF,Arch,ct_dim>::with_func( [&]( PI r, PI c ) {
        return cut_planes( ci[ r ], c );
    } );

    auto V = Vector<TF,Arch,ct_dim>::with_func( [&]( PI i ) {
        return cut_planes( ci[ i ], dim ) + add_func( ci[ i ] );
    } );

    return M.solve_ge( V );
}

UTP DTP::Pt DTP::solve_position( PI num_vertex ) const {
    return solve_position( num_vertex, []( auto ) { return 0; } );
}

// grow_infinite_cuts: parametrise by a scalar s such that INFINITE cut i gains s * norm_2( n_i )
// added to its offset (uniform spatial growth). sp(s) = cut_dir · v(s) - cut_dot is linear in s;
// evaluate at s=0 and s=1 via SimpleSquareMatrix::solve_ge. For each vertex with sp(0)<0 and
// sp(1)>0 the crossing is at s* = -sp(0)/(sp(1)-sp(0)). Apply s_grow = max(s*) to all INFINITE cuts.
UTP void DTP::grow_infinite_cuts( const auto &new_cut_dir, auto new_cut_dot ) {
    // check to grow enough so that all the vertices stay on the same side
    TF s_grow = 0; // std::numeric_limits<TF>::max();
    bool need_to_grow = false;
    for ( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex ) {
        if ( ! vertex_inf( num_vertex ) )
            continue;

        const Pt p1 = solve_position( num_vertex, [&]( PI num_cut ) { return cut_ids( num_cut ) == CellBoundary::INFINITE ? norm_2( cut_dir( num_cut ) ) : 0; } );
        const TF s0 = dot( vertex_position( num_vertex ), new_cut_dir ) - new_cut_dot;
        const TF s1 = dot( p1, new_cut_dir ) - new_cut_dot;

        // if it can cross 0
        const TF ds = s1 - s0;
        if ( ds && ( s0 > 0 ) != ( ds > 0 )  ) {
            const TF s_trial = - s0 / ( s1 - s0 );
            if ( s_grow < s_trial )
                s_grow = s_trial;
            need_to_grow = true;
            break;
        }
    }

    //
    if ( need_to_grow ) {
        s_grow += 1;

        // add s_grow * norm_2( n_c ) to each INFINITE cut's offset
        for ( PI num_cut = 0; num_cut < nb_cuts(); ++num_cut )
            if ( cut_ids( num_cut ) == CellBoundary::INFINITE )
                cut_planes( num_cut, dim ) += s_grow * norm_2( cut_dir( num_cut ) );

        // recompute vertex positions for all vertices with any INFINITE adjacent cut
        for ( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            if ( vertex_inf( num_vertex ) )
                vertex_positions.row( num_vertex ).get_data_from( solve_position( num_vertex ) );
    }
}

UTP void DTP::disp_cell() {
    info( nb_vertices() );
    for( PI i = 0; i < nb_vertices(); ++i ) {
        auto pos = Vector<TF,Arch,ct_dim>::with_func( 2, [&]( PI d ) { return vertex_positions( i, d ); } );
        auto cut = Vector<TF,Arch,ct_dim+1>::with_func( [&]( PI d ) { return cut_planes( i, d ); } );
        info( pos, cut );
    }
}

UTP void DTP::check_consistency() {
    auto get_cut_inds = [&]( PI v, PI *out ) {
        if ( dim == 2 ) {
            out[ 0 ] = ( v + nb_vertices - 1 ) % nb_vertices;
            out[ 1 ] = v;
        } else {
            for ( PI d = 0; d < dim; ++d )
                out[ d ] = vertex_indices( v, d );
        }
    };

    for( PI v = 0; v < nb_vertices(); ++v ) {
        PI ci[ ct_dim ];
        get_cut_inds( v, ci );

        auto M = Matrix<TF,Arch,ct_dim>::with_func( [&]( PI r, PI c ) { return cut_planes( ci[ r ], c ); } );
        auto V = Vector<TF,Arch,ct_dim>::with_func( [&]( PI i ) { return cut_planes( ci[ i ], dim ); } );
        const auto pos = M.solve_ge( V );

        for ( PI d = 0; d < dim; ++d )
            P( vertex_positions( v, d ), pos[ d ] );
    }
}

template<class Value,int ct_dim_value,class Arch,class TF,class TI>
struct Integral<Value,DTP> {
    static auto integral( const Value &value, DTP &cw ) {
        // infinite cell
        if ( ! cw.cell.is_fully_closed() )
            return std::numeric_limits<TF>::max();

        // simplex
        TF sum = 0;
        cw.for_each_simplex( [&]( const auto &simplex_indices ) {
            Simplex<ct_dim_value,ct_dim_value+1,TF,Arch> simplex;
            for( TI d = 0; d < cw.dim + 1; ++d )
                simplex.pts[ d ] = cw.vertex_positions.row( simplex_indices[ d ] );
            sum += sdot::integral( value, simplex );
        } );

        return sum;
    }
};

#undef UTP
#undef DTP

} // namespace sdot
