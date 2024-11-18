#include <sdot/distributions/ConstantValue.h>
#include <sdot/support/binding_config.h>
#include <tl/support/string/to_string.h>
 
#include <pybind11/pybind11.h>

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) {
    using namespace sdot;
    using TF = SDOT_CONFIG_scalar_type;

    // ConstantValue ---------------------------------------------------------------------------------------------------------------
    pybind11::class_<ConstantValue<TF>>( m, PD_STR( ConstantValue ) )
        .def( pybind11::init ( []( TF value ) -> ConstantValue<TF> { return { value }; } ) )
        .def( "__repr__"     , []( const ConstantValue<TF> &cv ) { return to_string( cv.value ); } )
        ;
}
