#pragma once

namespace sdot {

/** used by Mpi method to determine how the data can be sent/received */
struct MpiDataInfo  {
    bool assume_homogeneous_mpi_data_size = false; ///< used only if MpiContent<T>::size() is not ct known
};

} // namespace sdot
