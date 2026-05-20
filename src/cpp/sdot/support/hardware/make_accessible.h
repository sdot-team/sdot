#pragma once

namespace sdot {

void make_accessible( const auto &/* execution_space */, auto &&value, auto &&func ) {
    func( FORWARD( value ) );
}

} // namespace sdot


