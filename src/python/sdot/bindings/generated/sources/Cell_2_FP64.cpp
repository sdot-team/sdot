#line 30 "/Users/hugo.leclerc/sdot_with_interfaces/src/python/sdot/Cell.py"
#include <sdot/nanobind_wrappers.h>
#include <nanobind/stl/string.h>
#include <sdot/geometry/Cell.h>
#include <sstream>
#include <span>

namespace nb = nanobind;
using namespace sdot;

using NbArch = nanobind::device::cpu;
using Arch = ArchFor<NbArch>::type;
using TF = FP64;

using AF = nb::ndarray<const TF,NbArch>;
using MF = nb::ndarray<TF,NbArch>;

static constexpr int ct_dim = 2;
using CellType = Cell<TF,ct_dim,Arch>;
using Pt = CellType::Pt;

static Pt to_Pt( const auto &na ) {
    return std::span<const TF>( na.data(), na.size() );
}

NB_MODULE( Cell_2_FP64, m ) {
    nb::class_<CellType>( m, "Cell" )
        .def( "__init__", []( CellType *self, int dim ) {
            new ( self ) CellType(
                dim
            );
        } )
        .def( "__repr__", []( const CellType &b ) -> std::string {
            std::ostringstream ss;
            ss << b;
            return ss.str();
        } )
        .def( "cut", []( CellType &self, const AF dir, TF dot, PI id ) {
            self.cut( to_Pt( dir ), dot, id );
        } )
        // .def( "write_vtk", []( const CellType &b, std::string filename ) {
        //     VtkOutput vo;
        //     b.display_vtk( vo );
        //     vo.save( filename );
        // } )
    ;
}
