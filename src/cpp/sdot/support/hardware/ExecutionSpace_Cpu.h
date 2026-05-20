#pragma once

#include "ExecutionSpace.h"

namespace sdot {

// host execution / host RAM — stateless, {}-constructible
struct ExecutionSpace_Cpu : ExecutionSpace {
};

} // namespace sdot

