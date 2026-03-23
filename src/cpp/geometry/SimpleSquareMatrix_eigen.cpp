#include "SimpleSquareMatrix.h"
#include <Eigen/Eigenvalues>

namespace sdot {

template<class T,int ct_size>
auto SimpleSquareMatrix<T,ct_size>::eigen_system() const -> EigenSystem {
    const PI n = size();

    // copy lower triangular into Eigen matrix
    using EMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    EMat m( n, n );
    for( PI r = 0; r < n; ++r )
        for( PI c = 0; c < n; ++c )
            m( r, c ) = (*this)( r, c );

    Eigen::SelfAdjointEigenSolver<EMat> es( m.template selfadjointView<Eigen::Lower>() );

    // eigenvalues ascending, eigenvectors as rows
    EigenSystem res{ Point<T,ct_size>( n ), SimpleSquareMatrix<T,ct_size>( n ) };
    for( PI i = 0; i < n; ++i ) {
        res.values[ i ] = es.eigenvalues()[ i ];
        for( PI d = 0; d < n; ++d )
            res.vectors( i, d ) = es.eigenvectors().col( i )[ d ];
    }
    return res;
}

// Eigen compiled only in this translation unit
template auto SimpleSquareMatrix<FP32>::eigen_system() const -> SimpleSquareMatrix<FP32>::EigenSystem;
template auto SimpleSquareMatrix<FP64>::eigen_system() const -> SimpleSquareMatrix<FP64>::EigenSystem;

} // namespace sdot
