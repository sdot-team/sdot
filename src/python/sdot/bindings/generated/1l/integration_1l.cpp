#include <sdot/support/binding_config.h>
#include <sdot/Cell.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
using namespace sdot;
using Array_TF = pybind11::array_t<SDOT_CONFIG_scalar_type, pybind11::array::c_style>;
using Array_PI = pybind11::array_t<PI, pybind11::array::c_style>;
static constexpr int nb_dims = SDOT_CONFIG_nb_dims;
using Arch = sdot::SDOT_CONFIG_arch;
using TF = SDOT_CONFIG_scalar_type;
using Pt = Vec<TF, nb_dims>;
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
using TCell = Cell<Arch,TF,nb_dims,PD_NAME( CutInfo ),PD_NAME( CellInfo )>;
using TCut = Cut<TF,nb_dims,PD_NAME( CutInfo )>;
struct PD_NAME( Integration ) {
    void operator()( const Vec<Pt,nb_dims+1> &simplex ) {
        using namespace std;
        Eigen::Matrix<TF,nb_dims,nb_dims> M;
        for( PI r = 0; r < nb_dims; ++r )
            for( PI c = 0; c < nb_dims; ++c )
                M( r, c ) = simplex[ r + 1 ][ c ] - simplex[ 0 ][ c ];
        out[ 0 ] += M.determinant() / 2;
    }
    Vec<TF,1> out;
};
PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) {
    m.def( "cell_integral", []( TCell &cell, const std::vector<int> &rt_data ) -> std::vector<TF> {
        PD_NAME( Integration ) res;
        for( auto &v : res.out )
            v = 0;
        cell.simplex_split( res );
        return { res.out.begin(), res.out.end() };
    } );
}
