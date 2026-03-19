#pragma once

#include "PointFactory.h"
#include "VtkOutput.h"

namespace sdot {

//
template<class TF,int _dim=-1>
class Cell {
public:  
    using       PF                      = PointFactory<TF,_dim>;
    using       IF                      = PointFactory<PI,_dim>;
    using       VI                      = std::array<PI,2>;
    using       LK                      = std::vector<PI>;
    using       PT                      = Point<TF,_dim>;
    using       IT                      = Point<PI,_dim>;
                      
    struct      Vertex                  { PT pos; IT cut_indices; LK links; };
    struct      Cut                     { PT dir; TF sp; PI id; bool ext; };
                    
    using       Vertices                = std::vector<Vertex>;
    using       Cuts                    = std::vector<Cut>;
                      
    /**/        Cell                    ( int actual_dim, TF start_radius = 1 );

    void        display_vtk             ( VtkOutput &vo ) const;
    PI          dim                     () const { return pf.dim(); }

    static void for_each_2_comb_excepted( PI size, PI excepted, auto &&func );
    void        init_with_simplex       ( TF start_radius );
   
    Vertices    vertices;
    Cuts        cuts;
    PF          pf;
    PI          id;
};

//
// template<class T>
// class Cell<T,2> {
// public:

// };

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::Cell<T,d> &p );

#include "Cell.cxx"
