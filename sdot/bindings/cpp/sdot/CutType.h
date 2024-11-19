#pragma once

#include <tl/support/Displayer.h>

namespace sdot {

/**
 * @brief 
 *
 */
enum class CutType {
    Undefined,
    Boundary,
    Dirac,
};

inline void display( Displayer &ds, CutType ct ) {
    switch ( ct ) {
    case CutType::Undefined:  ds << "Undefined"; return;
    case CutType::Boundary:  ds << "Boundary"; return;
    case CutType::Dirac:  ds << "Dirac"; return;
    }
}

} // namespace sdot
