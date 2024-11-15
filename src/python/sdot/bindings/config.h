#pragma once

#ifndef SDOT_CONFIG_module_name
#define SDOT_CONFIG_module_name smurf
#endif

#ifndef SDOT_CONFIG_suffix
#define SDOT_CONFIG_suffix smurf
#endif

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

/// string with $SDOT_CONFIG_suffix
#define PD_STR( EXPR ) PD_STR_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )

