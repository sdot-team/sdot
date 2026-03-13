#include "../../src/cpu/sdot_w2_cpu.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>

namespace nb = nanobind;

// Bring C API functions into scope
using sdot::sdot_w2_cpu;
using sdot::sdot_w2_backward_cpu;

using ndarray_f32 = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ndarray_f32_2d = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Forward function that works with both 1D and 2D arrays
void sdot_w2_forward_impl(
    nb::ndarray<float> dirac_xs,
    nb::ndarray<float> dirac_ws,
    nb::ndarray<float> point_xs,
    nb::ndarray<float> point_ys,
    nb::ndarray<float> result,
    nb::ndarray<float> barycenters
) {
    size_t nb_diracs, nb_points;
    size_t batch_size = 1;

    // Determine dimensions
    if (dirac_xs.ndim() == 1) {
        batch_size = 1;
        nb_diracs = dirac_xs.shape(0);
    } else if (dirac_xs.ndim() == 2) {
        batch_size = dirac_xs.shape(0);
        nb_diracs = dirac_xs.shape(1);
    } else {
        throw std::invalid_argument("dirac_xs must be 1D or 2D");
    }

    if (point_xs.ndim() == 1) {
        nb_points = point_xs.shape(0);
        if (batch_size != 1) {
            throw std::invalid_argument("dirac_xs must have same batch size as point_xs");
        }
    } else if (point_xs.ndim() == 2) {
        nb_points = point_xs.shape(1);
        if (point_xs.shape(0) != batch_size) {
            throw std::invalid_argument("dirac_xs must have same batch size as point_xs");
        }
    } else {
        throw std::invalid_argument("point_xs must be 1D or 2D");
    }

    // Call CPU kernel
    sdot_w2_cpu(
        dirac_xs.data(), dirac_ws.data(), nb_diracs,
        point_xs.data(), point_ys.data(), nb_points,
        batch_size,
        result.data(), barycenters.data()
    );
}

void sdot_w2_backward_impl(
    nb::ndarray<float> grad_distance,
    nb::ndarray<float> grad_barycenters,
    nb::ndarray<float> w2_barycenters,
    nb::ndarray<float> dirac_xs,
    nb::ndarray<float> dirac_ws,
    nb::ndarray<float> points_xs,
    nb::ndarray<float> points_ys,
    nb::ndarray<float> grad_dirac_xs,
    nb::ndarray<float> grad_dirac_ws,
    nb::ndarray<float> grad_points_xs,
    nb::ndarray<float> grad_points_ys
) {
    size_t batch_size = 1;
    size_t Nf, Mg;

    if (dirac_xs.ndim() == 1) {
        batch_size = 1;
        Nf = dirac_xs.shape(0);
    } else if (dirac_xs.ndim() == 2) {
        batch_size = dirac_xs.shape(0);
        Nf = dirac_xs.shape(1);
    } else {
        throw std::invalid_argument("dirac_xs must be 1D or 2D");
    }

    if (points_xs.ndim() == 1) {
        Mg = points_xs.shape(0);
    } else if (points_xs.ndim() == 2) {
        Mg = points_xs.shape(1);
    } else {
        throw std::invalid_argument("points_xs must be 1D or 2D");
    }

    // Call CPU backward kernel
    sdot_w2_backward_cpu(
        grad_distance.data(),
        grad_barycenters.data(),
        w2_barycenters.data(),
        dirac_xs.data(),
        dirac_ws.data(),
        Nf,
        points_xs.data(),
        points_ys.data(),
        Mg,
        batch_size,
        grad_dirac_xs.data(),
        grad_dirac_ws.data(),
        grad_points_xs.data(),
        grad_points_ys.data()
    );
}

NB_MODULE(sdot_pytorch_cpp, m) {
    m.def("forward_impl", &sdot_w2_forward_impl,
          nb::arg("dirac_xs"), nb::arg("dirac_ws"),
          nb::arg("point_xs"), nb::arg("point_ys"),
          nb::arg("result"), nb::arg("barycenters"),
          "SDOT W2 forward implementation");

    m.def("backward_impl", &sdot_w2_backward_impl,
          nb::arg("grad_distance"), nb::arg("grad_barycenters"),
          nb::arg("w2_barycenters"),
          nb::arg("dirac_xs"), nb::arg("dirac_ws"),
          nb::arg("points_xs"), nb::arg("points_ys"),
          nb::arg("grad_dirac_xs"), nb::arg("grad_dirac_ws"),
          nb::arg("grad_points_xs"), nb::arg("grad_points_ys"),
          "SDOT W2 backward implementation");
}
