#pragma once

// helper for PD_NAME
#define PD_CONCAT_MACRO_( A, B ) A##_##B
#define PD_CONCAT_MACRO( A, B ) PD_CONCAT_MACRO_( A, B)

#define PD_STR_CONCAT_MACRO_( A, B ) #A "_" #B
#define PD_STR_CONCAT_MACRO( A, B ) PD_STR_CONCAT_MACRO_( A, B )

/// concatenation with $SDOT_CONFIG_suffix
#define PD_NAME( EXPR ) PD_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )

///
#define PD_STR( EXPR ) PD_STR_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )
