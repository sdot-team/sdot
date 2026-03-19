#pragma once

#include "MapOfUniqueSortedIndices.h"
#include "PointFactory.h"
#include "VtkOutput.h"

namespace sdot {

//
template<class TF,int _dim=-1>
class Cell {
public:  
    using       EdgeMap                 = MapOfUniqueSortedIndices<(_dim>0?_dim-1:-1)>;
    using       FaceMap                 = MapOfUniqueSortedIndices<(_dim>1?_dim-2:-1)>;
    using       PF                      = PointFactory<TF,_dim>;
    using       IF                      = PointFactory<PI,_dim>;
    using       VF                      = std::vector<TF>;
    using       Pt                      = Point<TF,_dim>;
    using       It                      = Point<PI,_dim>;
                      
    struct      EdgeLink                { PI vertex_index; PI num_cut_to_remove; };
    struct      Vertex                  { Pt pos; It cut_indices; std::vector<EdgeLink> edge_links; TF s; PI op_id = 0; };
    struct      Cut                     { Pt dir; TF sp; PI id; bool ext; };
                    
    using       Vertices                = std::vector<Vertex>;
    using       Cuts                    = std::vector<Cut>;
                      
    /**/        Cell                    ( int actual_dim, TF start_radius = 1 );

    void        display_vtk             ( VtkOutput &vo ) const;
    PI          dim                     () const { return pf.dim(); }

    void        cut                     ( const Pt &dir, TF sp, PI id );

    static void for_each_2_comb_excepted( PI size, PI excepted, auto &&func );
    void        init_with_simplex       ( TF start_radius );

    PI          curr_op_id;             ///<
    Vertices    vertices;               ///<
    Cuts        cuts;                   ///<
    VF          sps;                    ///< scalar product
    PF          pf;                     ///< point factory
    PI          id;                     ///< index factory
};

//
// template<class T>
// class Cell<T,2> {
// public:

// };

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::Cell<T,d> &p );

#include "Cell.cxx"
