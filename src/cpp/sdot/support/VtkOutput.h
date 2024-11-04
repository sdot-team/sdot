#pragma once

#include <tl/support/containers/Vec.h>
#include <map>

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

    using           TF                   = double;
    using           Pt                   = Vec<TF,3>;
    using           VTF                  = Vec<TF>;
    using           FieldMap             = std::map<std::string,VTF>;

    /**/            VtkOutput            ();

    void            save                 ( std::string filename ) const;
    void            save                 ( std::ostream &os ) const;

    // fixed #pts
    void            add_triangle         ( Vec<Pt,3> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_pyramid          ( Vec<Pt,5> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_wedge            ( Vec<Pt,6> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_tetra            ( Vec<Pt,4> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_point            ( Pt pt, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_hexa             ( Vec<Pt,8> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_quad             ( Vec<Pt,4> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_edge             ( Vec<Pt,2> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );

    // variable #pts
    void            add_polygon          ( Span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );
    void            add_line             ( Span<Pt> pts, const std::map<std::string,VTF> &point_data = {}, const std::map<std::string,TF> &cell_data = {} );

    // generic
    void            add_item             ( const Pt *pts_data, PI pts_size, PI vtk_type, const std::map<std::string,VTF> &point_data, const std::map<std::string,TF> &cell_data );

    // type info
    // static void  get_compilation_flags( Vfs::CompilationFlags &cn ) { cn.add_inc_file( "sdot/VtkOutput.h" ); }
    static auto     type_name            () { return "VtkOutput"; }

    FieldMap        point_fields;        ///<
    FieldMap        cell_fields;         ///<
    Vec<PI>         cell_types;
    Vec<PI>         cell_items;
    Vec<Pt>         points;
};

