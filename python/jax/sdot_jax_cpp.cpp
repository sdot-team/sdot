#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "../../src/cpu/sdot_w2_cpu.h"
#include "../../src/cpu/sdot_l2_cpu.h"
#include <vector>
#include <cstdint>

namespace nb = nanobind;

void sdot_l2_cpu_wrapped(nb::ndarray<float, nb::device::cpu> f, 
                         nb::ndarray<float, nb::device::cpu> g,
                         nb::ndarray<float, nb::device::cpu> result) {
    sdot_l2_cpu(f.data(), g.data(), f.shape(0), result.data());
}

void sdot_l2_backward_cpu_wrapped(nb::ndarray<float, nb::device::cpu> f, 
                                  nb::ndarray<float, nb::device::cpu> g,
                                  nb::ndarray<float, nb::device::cpu> grad_f,
                                  nb::ndarray<float, nb::device::cpu> grad_g) {
    sdot_l2_backward_cpu(f.data(), g.data(), f.shape(0), grad_f.data(), grad_g.data());
}

void sdot_w2_cpu_wrapped(nb::ndarray<float, nb::device::cpu> Xf, 
                         nb::ndarray<float, nb::device::cpu> Wf,
                         nb::ndarray<float, nb::device::cpu> Xg,
                         nb::ndarray<float, nb::device::cpu> Yg,
                         nb::ndarray<float, nb::device::cpu> result,
                         nb::ndarray<float, nb::device::cpu> barycenters) {
    
    size_t batch_size = 1;
    size_t Nf, Mg;
    
    if (Xf.ndim() == 1) {
        batch_size = 1;
        Nf = Xf.shape(0);
    } else {
        batch_size = Xf.shape(0);
        Nf = Xf.shape(1);
    }
    
    if (Xg.ndim() == 1) {
        Mg = Xg.shape(0);
    } else {
        Mg = Xg.shape(1);
    }
    
    sdot_w2_cpu(Xf.data(), Wf.data(), Nf,
                Xg.data(), Yg.data(), Mg,
                batch_size,
                result.data(), barycenters.data());
}

NB_MODULE(sdot_jax_cpp, m) {
    m.def("w2_cpu", &sdot_w2_cpu_wrapped);
    m.def("l2_cpu", &sdot_l2_cpu_wrapped);
    m.def("l2_backward_cpu", &sdot_l2_backward_cpu_wrapped);
}
