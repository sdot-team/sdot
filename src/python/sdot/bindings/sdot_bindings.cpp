// #include <sdot/DiracVecFromLocallyKnownValues.h>
// #include <sdot/HomogeneousWeights.h>
// #include <sdot/WeightsWithBounds.h>
// #include <sdot/PavingWithDiracs.h>
// #include <sdot/RegularGrid.h>
#include "sdot/local_weight_bounds/LocalWeightBounds.h"
#include <sdot/local_weight_bounds/LocalWeightBounds_ConstantValue.h>
#include <sdot/vec_readers/KnownVecReader.h>
#include <sdot/pavings/RegularGrid.h>
#include <sdot/support/VtkOutput.h>
#include <sdot/Cell.h>

#include <tl/support/string/to_string.h>
 
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include "pybind11/gil.h"
 
#ifndef SDOT_CONFIG_scalar_type
#define SDOT_CONFIG_scalar_type FP64
#endif

#ifndef SDOT_CONFIG_nb_dims
#define SDOT_CONFIG_nb_dims 3
#endif

#ifndef SDOT_CONFIG_arch
#define SDOT_CONFIG_arch YoArch
#endif

// helper for PD_NAME
#define PD_CONCAT_MACRO_( A, B ) A##_##B
#define PD_CONCAT_MACRO( A, B ) PD_CONCAT_MACRO_( A, B)

#define PD_STR_CONCAT_MACRO_( A, B ) #A "_" #B
#define PD_STR_CONCAT_MACRO( A, B ) PD_STR_CONCAT_MACRO_( A, B )

/// concatenation with $SDOT_CONFIG_suffix
#define PD_NAME( EXPR ) PD_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )

///
#define PD_STR( EXPR ) PD_STR_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )


// arch
namespace sdot { struct SDOT_CONFIG_arch {}; }

static constexpr int nb_dims = SDOT_CONFIG_nb_dims;
using TF = SDOT_CONFIG_scalar_type;
using Arch = sdot::SDOT_CONFIG_arch;

using Array = pybind11::array_t<TF, pybind11::array::c_style>;
using Pt = Vec<TF, nb_dims>;
using namespace sdot;

struct PD_NAME( CutInfo ) {
    CutType type;
    Pt      p1;
    TF      w1;
    PI      i1;
};

struct PD_NAME( CellInfo ) {
    Pt p0;
    TF w0;
    PI i0;
};

// #define PolyCon_Py CC_DT( PolyCon_py )

// Ti static Array to_Array( const Vec<Scalar,i> &v ) {
//     Vec<PI,1> shape{ v.size() };
//     Array res( shape );
//     for( PI n = 0; n < PI( v.size() ); ++n )
//         res.mutable_at( n ) = v[ n ];
//     return res;
// }


// static Array to_Array( const Vec<Point> &v, const Vec<Scalar> &s ) {
//     Vec<PI,2> shape{ v.size(), PI( POLYCON_DIM + 1 ) };
//     Array res( shape );
//     for( PI i = 0; i < v.size(); ++i ) {
//         for( PI d = 0; d < PI( POLYCON_DIM ); ++d )
//             res.mutable_at( i, d ) = v[ i ][ d ];
//         res.mutable_at( i, POLYCON_DIM ) = s[ i ];
//     }
//     return res;
// }

// /***/
// struct PolyCon_py {
//     PolyCon_py( Array a_dir, Array a_off, Array b_dir, Array b_off ) : pc(
//             Span<Point>{ reinterpret_cast<Point *>( a_dir.mutable_data() ), PI( a_dir.shape( 0 ) ) },
//             { a_off.mutable_data(), PI( a_off.shape( 0 ) ) },
//             Span<Point>{ reinterpret_cast<Point *>( b_dir.mutable_data() ), PI( b_dir.shape( 0 ) ) },
//             { b_off.mutable_data(), PI( b_off.shape( 0 ) ) }
//     ) { }

//     PolyCon_py( PolyCon<Scalar,POLYCON_DIM> &&pc ) : pc( std::move( pc ) ) {
//     }

//     void write_vtk( const Str &filename ) {
//         VtkOutput vo;
//         pc.display_vtk( vo );
//         vo.save( filename );
//     }

//     std::variant<std::tuple<Scalar,Array>, py::none> value_and_gradient( Array x ) {
//         Point p( FromItemValue(), 0 );
//         for( PI i = 0; i < std::min( PI( POLYCON_DIM ), PI( x.size() ) ); ++i )
//             p[ i ] = x.at( i );
         
//         if ( auto vg = pc.value_and_gradient( p ) )
//             return std::tuple<Scalar,Array>{ std::get<0>( *vg ), to_Array( std::get<1>( *vg ) ) };
//         return py::none();
//     }

//     std::variant<std::tuple<Scalar,std::vector<Array>>, py::none> value_and_gradients( Array x, Scalar probe_size ) {
//         Point p( FromItemValue(), 0 );
//         for( PI i = 0; i < std::min( PI( POLYCON_DIM ), PI( x.size() ) ); ++i )
//             p[ i ] = x.at( i );
         
//         if ( auto vg = pc.value_and_gradients( p, probe_size ) ) {
//             std::vector<Array> gradients;
//             for( const auto &g : std::get<1>( *vg ) )
//                 gradients.push_back( to_Array( g ) );
//             return std::tuple<Scalar,std::vector<Array>>{ std::get<0>( *vg ), gradients };
//         }
//         return py::none();
//     }

//     PolyCon_py legendre_transform() {
//         return pc.legendre_transform();
//     }

//     auto as_fbdo_arrays() -> std::tuple<Array,Array,Array,Array> {
//         return { to_Array( pc.f_dirs ), to_Array( pc.f_offs ), to_Array( pc.b_dirs ), to_Array( pc.b_offs ) };
//     }

//     auto as_fb_arrays() -> std::tuple<Array,Array> {
//         return { to_Array( pc.f_dirs, pc.f_offs ), to_Array( pc.b_dirs, pc.b_offs ) };
//     }

//     struct VertexData {
//         Scalar          height;
//         CountOfCutTypes cct;
//         Point           pos;
//     };

//     auto edge_data( CtInt<1> ) {
//         Vec<Vec<VertexData,2>> res;
//         pc.for_each_cell( [&]( Cell<Scalar,POLYCON_DIM> &cell ) {
//             cell.for_each_edge( [&]( auto num_cuts, const Vertex<Scalar,POLYCON_DIM> &v0, const Vertex<Scalar,POLYCON_DIM> &v1 ) {
//                 const Scalar h0 = cell.height( v0.pos );
//                 const Scalar h1 = cell.height( v1.pos );

//                 CountOfCutTypes c0, c1;
//                 cell.add_cut_types( c0, v0.num_cuts, pc.nb_bnds() );
//                 cell.add_cut_types( c1, v1.num_cuts, pc.nb_bnds() );

//                 res << Vec<VertexData,2> {
//                     VertexData{ h0, c0, v0.pos },
//                     VertexData{ h1, c1, v1.pos },
//                 };
//             } );
//         } );
//         return res;
//     }

//     template<int nd>
//     auto edge_data( CtInt<nd> ) {
//         using NC = Vec<SI,POLYCON_DIM-1>;
//         std::map<NC,Vec<VertexData,2>,Less> map;
//         pc.for_each_cell( [&]( Cell<Scalar,POLYCON_DIM> &cell ) {
//             cell.for_each_edge( [&]( auto num_cuts, const Vertex<Scalar,POLYCON_DIM> &v0, const Vertex<Scalar,POLYCON_DIM> &v1 ) {
//                 std::sort( num_cuts.begin(), num_cuts.end() );
//                 map_item( map, num_cuts, [&]() -> Vec<VertexData,2> {
//                     const Scalar h0 = cell.height( v0.pos );
//                     const Scalar h1 = cell.height( v1.pos );

//                     CountOfCutTypes c0, c1;
//                     cell.add_cut_types( c0, v0.num_cuts, pc.nb_bnds() );
//                     cell.add_cut_types( c1, v1.num_cuts, pc.nb_bnds() );

//                     return {
//                         VertexData{ h0, c0, v0.pos },
//                         VertexData{ h1, c1, v1.pos },
//                     };
//                 } );
//             } );
//         } );

//         Vec<Vec<VertexData,2>> res;
//         for( const auto &p : map )
//             res << p.second;
//         return res;
//     }

//     Array edge_points() {
//         auto map = edge_data( CtInt<POLYCON_DIM>() );

//         Vec<PI> shape{ map.size(), 2, POLYCON_DIM + 4 };
//         Array res( shape );
//         for( PI n = 0; n < map.size(); ++n ) {
//             for( PI i = 0; i < 2; ++i ) {
//                 for( PI d = 0; d < POLYCON_DIM; ++d )
//                     res.mutable_at( n, i, d ) = map[ n ][ i ].pos[ d ];
//                 res.mutable_at( n, i, POLYCON_DIM + 0 ) = map[ n ][ i ].height;
//                 res.mutable_at( n, i, POLYCON_DIM + 1 ) = map[ n ][ i ].cct.nb_ints;
//                 res.mutable_at( n, i, POLYCON_DIM + 2 ) = map[ n ][ i ].cct.nb_bnds;
//                 res.mutable_at( n, i, POLYCON_DIM + 3 ) = map[ n ][ i ].cct.nb_infs;
//             }
//         }
//         return res;
//     }

//     PI ndim() const {
//         return pc.ndim();
//     }

//     PolyCon_py add_polycon( const PolyCon_py &that ) {
//         return pc + that.pc;
//     }

//     PolyCon_py add_scalar( Scalar that ) {
//         Vec<Scalar> new_f_offs = pc.f_offs - that;
//         return PolyCon<Scalar,POLYCON_DIM>( pc.f_dirs, new_f_offs, pc.b_dirs, pc.b_offs );
//     }

//     PolyCon_py mul_scalar( Scalar that ) {
//         Vec<Point>  new_f_dirs = that * pc.f_dirs;
//         Vec<Scalar> new_f_offs = that * pc.f_offs;
//         return PolyCon<Scalar,POLYCON_DIM>( new_f_dirs, new_f_offs, pc.b_dirs, pc.b_offs );
//     }

//     PolyCon_py normalized( POLYCON_SCALAR min_volume = 0 ) {
//         PolyCon<POLYCON_SCALAR,POLYCON_DIM> cp = pc;
//         cp.normalize( min_volume );
//         return cp;
//     }

//     PolyCon<POLYCON_SCALAR,POLYCON_DIM> pc;
// };

Pt Pt_from_Array( const Array &array ) {
    Pt res;
    if ( array.size() < nb_dims )
        throw pybind11::value_error( "array is not large enough" );
    for( PI d = 0; d < nb_dims; ++d )
       res[ d ] = array.at( d );
    return res;
}

Vec<Pt> VecPt_from_Array( const Array &array ) {
    Vec<Pt> res;
    if ( array.shape( 1 ) < nb_dims )
        throw pybind11::value_error( "array is not large enough" );
    res.resize( array.shape( 0 ) );
    for( PI r = 0; r < res.size(); ++r )
        for( PI d = 0; d < nb_dims; ++d )
           res[ r ][ d ] = array.at( r, d );
    return res;
}

template<class T,int s>
static auto Array_from_VecPt( const Vec<Vec<T,s>> &v ) {
    Vec<PI,2> shape{ v.size(), PI( s ) };
    pybind11::array_t<T, pybind11::array::c_style> res( shape );
    for( PI i = 0; i < v.size(); ++i )
        for( PI d = 0; d < s; ++d )
            res.mutable_at( i, d ) = v[ i ][ d ];
    return res;
}

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    // pybind11::class_<Pt>( m, PD_STR( Pt ) )
    //     .def( pybind11::init( &create_Pt_from_Array ) )
    //     .def( "__repr__", []( const Pt &pt ) { return to_string( pt ); } )
    //     ;
    using TCell = Cell<Arch,TF,nb_dims,PD_NAME( CutInfo ),PD_NAME( CellInfo )>;
 
    pybind11::class_<VtkOutput>( m, PD_STR( VtkOutput ) )
        .def( pybind11::init<>() )
        .def( "save", []( VtkOutput &vo, Str fn ) { return vo.save( fn ); } )
        ;
 
    pybind11::class_<TCell>( m, PD_STR( Cell ) )
        // base methods
        .def( pybind11::init<>() )

        // modifications
        .def( "cut_boundary", []( TCell &cell, const Array &dir, TF off, PI ind ) { return cell.cut( Pt_from_Array( dir ), off, { .type = CutType::Boundary, .i1 = ind } ); } )
        .def( "cut", []( TCell &cell, const Array &dir, TF off, PI ind ) { return cell.cut( Pt_from_Array( dir ), off, { .type = CutType::Undefined, .i1 = ind } ); } )
        
        // properties
        .def( "true_dimensionality", &TCell::true_dimensionality )
        .def( "nb_active_cuts", &TCell::nb_active_cuts )
        .def( "nb_stored_cuts", &TCell::nb_stored_cuts )
        .def( "nb_vertices", []( TCell &cell, bool current_dim ) { return current_dim ? cell.nb_vertices_true_dim() : cell.nb_vertices(); } )
        .def( "bounded", &TCell::bounded )
        .def( "closed_faces", []( const TCell &cell ) {
            std::vector<std::vector<PI>> faces;
            cell.for_each_closed_face( [&]( auto &&refs, auto &&vertices ) {
                faces.push_back( { vertices.begin(), vertices.end() } );
            } );
            return faces;
        } )
        .def( "empty", &TCell::empty )
        .def( "ndim", []( const TCell &cell ) { return nb_dims; } )
        .def( "base", []( TCell &cell ) { return Array_from_VecPt( cell.base() ); } )

        .def( "vertex_coords", []( const TCell &cell, bool allow_lower_dim ) {
            if ( allow_lower_dim ) {
                return cell.with_ct_dim( [&]( auto td ) {
                    Vec<Vec<TF,td>> coords;
                    cell.for_each_vertex_coord( [&]( const auto &pt ) {
                        coords << pt;
                    }, td );
                    return Array_from_VecPt( coords );
                } );
            }

            Vec<Pt> coords;
            cell.for_each_vertex_coord( [&]( const auto &pt ) {
                coords << pt;
            } );
            return Array_from_VecPt( coords );
        } )

        .def( "vertex_refs", []( const TCell &cell, bool allow_lower_dim ) {
            if ( allow_lower_dim ) {
                return cell.with_ct_dim( [&]( auto td ) {
                    Vec<Vec<PI,td>> refs;
                    cell.for_each_vertex_ref( [&]( const auto &pt ) {
                        refs << pt;
                    }, td );
                    return Array_from_VecPt( refs );
                } );
            }

            Vec<Vec<PI,nb_dims>> refs;
            cell.for_each_vertex_ref( [&]( const auto &pt ) {
                refs << pt;
            } );
            return Array_from_VecPt( refs );
        } )

        // // output
        .def( "display_vtk", []( const TCell &cell, VtkOutput &vo ) { return cell.display_vtk( vo ); } )
        .def( "__repr__", []( const TCell &cell ) { return to_string( cell ); } )
        ;

    pybind11::class_<KnownVecReader<Pt>>( m, PD_STR( KnownVecOfPointsReader ) )
        .def( pybind11::init( [&]( const Array &pts ) -> KnownVecReader<Pt> { return VecPt_from_Array( pts ); } ) )
        .def( "__repr__", []( const KnownVecReader<Pt> &a ) { return to_string( a ); } )
        ;

    // // weights ======================================================================================
    pybind11::class_<LocalWeightBounds<TCell>>( m, PD_STR( LocalWeightBounds ) )
        .def( "__repr__", []( const LocalWeightBounds<TCell> &a ) { return to_string( a ); } )
        ;

    pybind11::class_<LocalWeightBounds_ConstantValue<TCell>,LocalWeightBounds<TCell>>( m, PD_STR( LocalWeightBounds_ConstantValue ) )
        .def( pybind11::init<TF>() )
        // .def( "__repr__", []( const HomogeneousWeights &a ) { return to_string( a ); } )
        ;

    // paving ========================================================================================
    // pybind11::class_<PavingWithDiracs>( m, PD_STR( PavingWithDiracs ) )
    //     .def( "__repr__", []( const PavingWithDiracs &a ) { return to_string( a ); } )
    //     ;

    pybind11::class_<RegularGrid<TCell>>( m, PD_STR( RegularGrid ) )
        .def( pybind11::init( []( const KnownVecReader<Pt> &pts ) -> RegularGrid<TCell> { return { pts, {} }; } ) )
        .def( "__repr__", []( const RegularGrid<TCell> &a ) { return to_string( a ); } )
        .def( "for_each_cell", []( RegularGrid<TCell> &paving, const TCell &base_cell, const LocalWeightBounds<TCell> &wwb, const std::function<void( const TCell &cell )> &f, int max_nb_threads ) { 
            pybind11::gil_scoped_release gsr;
            paving.for_each_cell( base_cell, wwb, [&]( TCell &cell, int ) {
                pybind11::gil_scoped_acquire gsa;
                f( cell );
            }, max_nb_threads );
        } )
        ;

    // // utility functions ============================================================================
    // m.def( "display_vtk", []( VtkOutput &vo, const Cell &base_cell, PavingWithDiracs &diracs, const WeightsWithBounds &weights ) {
    //     std::mutex m;
    //     diracs.for_each_cell( base_cell, weights, [&]( Cell &cell, int ) { 
    //         m.lock();
    //         cell.display_vtk( vo );
    //         m.unlock();
    //     } );
    // } );
    
    m.def( "ndim", []() {
        return nb_dims;
    } );
}
