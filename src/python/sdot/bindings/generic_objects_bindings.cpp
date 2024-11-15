// #include <tl/support/type_info/type_name.h>
// #include <tl/support/string/to_string.h>
#include <sdot/support/VtkOutput.h>
#include "config.h"
 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace sdot;

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    pybind11::class_<VtkOutput>( m, "VtkOutput" )
        .def( pybind11::init<>() )
        .def( "save", []( VtkOutput &self, Str filename ) { self.save( filename ); } )
        ;
}
