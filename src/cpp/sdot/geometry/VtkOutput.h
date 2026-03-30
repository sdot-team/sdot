#pragma once

#include <vector>
#include <string>
#include <array>
#include <span>
#include <map>

namespace sdot {

/***/
class VtkOutput {
public:
    enum {          VtkTriangle          = 5   };
    enum {          VtkPyramid           = 14  };
    enum {          VtkPolygon           = 7   };
    enum {          VtkWedge             = 13  };
    enum {          VtkPoint             = 1   };
    enum {          VtkTetra             = 10  };
    enum {          VtkHexa              = 12  };
    enum {          VtkLine              = 4   };
    enum {          VtkPoly              = 7   };
    enum {          VtkQuad              = 9   };

    using           PI                   = std::size_t;
    using           TF                   = double;
    using           Pt                   = std::array<TF,3>;
    using           VTF                  = std::vector<TF>;
    using           FieldMap             = std::map<std::string,VTF>;

    /**/            VtkOutput            ();

    void            save                 ( std::string filename ) const;
    void            save                 ( std::ostream &os ) const;

    // fixed #pts
    void            add_triangle         ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_pyramid          ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_wedge            ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_tetra            ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_hexa             ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_quad             ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_edge             ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );

    void            add_point            ( Pt pt, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );

    // variable #pts
    void            add_polygon          ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_line             ( std::span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );

    // generic
    void            add_item             ( const Pt *pts_data, PI pts_size, PI vtk_type, const std::map<std::string,VTF> &point_data, const std::map<std::string,TF> &cell_data );

    // type info
    // static void  get_compilation_flags( Vfs::CompilationFlags &cn ) { cn.add_inc_file( "sdot/VtkOutput.h" ); }
    static auto     type_name            () { return "VtkOutput"; }

    FieldMap        point_fields;        ///<
    FieldMap        cell_fields;         ///<
    std::vector<PI> cell_types;
    std::vector<PI> cell_items;
    std::vector<Pt> points;
};

}
