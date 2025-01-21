#include <sdot/symbolic/instructions/SymbolicArray.h>
#include <sdot/support/binding_config.h>
#include <sdot/symbolic/Expr.h>

#ifndef SDOT_CONFIG_interpolation_order
#define SDOT_CONFIG_interpolation_order 0
#endif

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) {
    using namespace sdot;
    using TF = SDOT_CONFIG_scalar_type;
    static constexpr int nb_dims = SDOT_CONFIG_nb_dims;
    static constexpr int interpolation_order = SDOT_CONFIG_interpolation_order;
    using Array_TF = pybind11::array_t<TF, pybind11::array::c_style>;

    // Expr_from_image ---------------------------------------------------------------------------------------------------------------
    m.def( "symbolic_array", []( const Array_TF &array, const std::vector<Expr> &indices ) {
        auto *res = new SymbolicArray<TF,nb_dims>;
        
        for( PI d = 0; d < nb_dims; ++d )
            res->extents[ d ] = array.shape( d );
        
        res->values.resize( res->nb_values() );
        for( PI v = 0, n = res->values.size(); v < n; ++v )
            res->values[ v ] = array.data()[ v ];

        // for( PI r = 0; r < nb_dims; ++r )
        //     for( PI c = 0; c < nb_dims; ++c )
        //         res->trinv.linear_transformation[ r ][ c ] = trinv.at( r, c );
        // for( PI d = 0; d < nb_dims; ++d )
        //     res->trinv.translation[ d ] = trinv.at( d, nb_dims );

        for( const Expr &ind : indices )
            res->add_child( ind.inst );

        return Expr{ res };
    } );
}
