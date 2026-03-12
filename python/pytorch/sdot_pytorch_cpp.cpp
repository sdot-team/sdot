#include "../../src/cpu/sdot_w2_cpu.h"
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> sdot_w2_forward(torch::Tensor dirac_xs, torch::Tensor dirac_ws, torch::Tensor point_xs, torch::Tensor point_ys) {
    int64_t batch_size = 1;
    int64_t Nf, Mg;

    if (dirac_xs.dim() == 1) {
        batch_size = 1;
        Nf = dirac_xs.size(0);
    } else {
        batch_size = dirac_xs.size(0);
        Nf = dirac_xs.size(1);
    }

    if (point_xs.dim() == 1) {
        Mg = point_xs.size(0);
        TORCH_CHECK(batch_size == 1, "Xg must have same batch size as Xf");
    } else {
        Mg = point_xs.size(1);
        TORCH_CHECK(point_xs.size(0) == batch_size, "Xg must have same batch size as Xf");
    }

    // Check Wf and Yg consistency
    TORCH_CHECK(dirac_ws.sizes() == dirac_xs.sizes(), "Wf and Xf must have same shapes");
    TORCH_CHECK(point_ys.sizes() == point_xs.sizes(), "Yg and Xg must have same shapes");

    auto result_shape = (dirac_xs.dim() == 1) ? std::vector<int64_t>({1}) : std::vector<int64_t>({batch_size});
    auto result = torch::empty(result_shape, dirac_xs.options());
    auto barycenters = torch::empty(dirac_xs.sizes(), dirac_xs.options());

    if (dirac_xs.device().is_cpu()) {
        sdot_w2_cpu(dirac_xs.data_ptr<float>(), dirac_ws.data_ptr<float>(), Nf,
                    point_xs.data_ptr<float>(), point_ys.data_ptr<float>(), Mg,
                    batch_size,
                    result.data_ptr<float>(), barycenters.data_ptr<float>());
    } else {
        auto Xf_cpu = dirac_xs.to(torch::kCPU).contiguous();
        auto Wf_cpu = dirac_ws.to(torch::kCPU).contiguous();
        auto Xg_cpu = point_xs.to(torch::kCPU).contiguous();
        auto Yg_cpu = point_ys.to(torch::kCPU).contiguous();

        std::vector<float> res_vals(batch_size);
        std::vector<float> bary_vals(batch_size * Nf);

        sdot_w2_cpu(Xf_cpu.data_ptr<float>(), Wf_cpu.data_ptr<float>(), Nf,
                    Xg_cpu.data_ptr<float>(), Yg_cpu.data_ptr<float>(), Mg,
                    batch_size,
                    res_vals.data(), bary_vals.data());

        result.copy_(torch::from_blob(res_vals.data(), result_shape, torch::kFloat32));
        barycenters.copy_(torch::from_blob(bary_vals.data(), dirac_xs.sizes(), torch::kFloat32));
    }

    return {result, barycenters};
}

std::vector<torch::Tensor> sdot_w2_backward(
    torch::Tensor grad_distance,
    torch::Tensor grad_barycenters,
    torch::Tensor w2_barycenters,
    torch::Tensor dirac_xs, torch::Tensor dirac_ws,
    torch::Tensor points_xs, torch::Tensor points_ys
) {
    int64_t batch_size = 1;
    int64_t Nf, Mg;

    if (dirac_xs.dim() == 1) {
        batch_size = 1;
        Nf = dirac_xs.size(0);
    } else {
        batch_size = dirac_xs.size(0);
        Nf = dirac_xs.size(1);
    }

    if (points_xs.dim() == 1) {
        Mg = points_xs.size(0);
    } else {
        Mg = points_xs.size(1);
    }

    auto grad_dirac_xs = torch::empty_like(dirac_xs);
    auto grad_dirac_ws = torch::empty_like(dirac_ws);
    auto grad_points_xs = torch::empty_like(points_xs);
    auto grad_points_ys = torch::empty_like(points_ys);

    if (dirac_xs.device().is_cpu()) {
        sdot_w2_backward_cpu(
            grad_distance.contiguous().data_ptr<float>(),
            grad_barycenters.contiguous().data_ptr<float>(),
            w2_barycenters.contiguous().data_ptr<float>(),
            dirac_xs.contiguous().data_ptr<float>(),
            dirac_ws.contiguous().data_ptr<float>(),
            Nf,
            points_xs.contiguous().data_ptr<float>(),
            points_ys.contiguous().data_ptr<float>(),
            Mg,
            batch_size,
            grad_dirac_xs.data_ptr<float>(),
            grad_dirac_ws.data_ptr<float>(),
            grad_points_xs.data_ptr<float>(),
            grad_points_ys.data_ptr<float>()
        );
    } else {
        auto gdist_cpu = grad_distance.to(torch::kCPU).contiguous();
        auto gbary_cpu = grad_barycenters.to(torch::kCPU).contiguous();
        auto wbary_cpu = w2_barycenters.to(torch::kCPU).contiguous();
        auto dxs_cpu = dirac_xs.to(torch::kCPU).contiguous();
        auto dws_cpu = dirac_ws.to(torch::kCPU).contiguous();
        auto pxs_cpu = points_xs.to(torch::kCPU).contiguous();
        auto pys_cpu = points_ys.to(torch::kCPU).contiguous();

        auto gdxs_cpu = torch::empty_like(dxs_cpu);
        auto gdws_cpu = torch::empty_like(dws_cpu);
        auto gpxs_cpu = torch::empty_like(pxs_cpu);
        auto gpys_cpu = torch::empty_like(pys_cpu);

        sdot_w2_backward_cpu(
            gdist_cpu.data_ptr<float>(),
            gbary_cpu.data_ptr<float>(),
            wbary_cpu.data_ptr<float>(),
            dxs_cpu.data_ptr<float>(),
            dws_cpu.data_ptr<float>(),
            Nf,
            pxs_cpu.data_ptr<float>(),
            pys_cpu.data_ptr<float>(),
            Mg,
            batch_size,
            gdxs_cpu.data_ptr<float>(),
            gdws_cpu.data_ptr<float>(),
            gpxs_cpu.data_ptr<float>(),
            gpys_cpu.data_ptr<float>()
        );

        grad_dirac_xs.copy_(gdxs_cpu);
        grad_dirac_ws.copy_(gdws_cpu);
        grad_points_xs.copy_(gpxs_cpu);
        grad_points_ys.copy_(gpys_cpu);
    }

    return {grad_dirac_xs, grad_dirac_ws, grad_points_xs, grad_points_ys};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdot_w2_forward, "SDOT W2 forward");
    m.def("backward", &sdot_w2_backward, "SDOT W2 backward");
}
