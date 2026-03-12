#include <torch/extension.h>
#include "../../src/cpu/sdot_w2_cpu.h"
#include "../../src/cpu/sdot_l2_cpu.h"
#include <vector>

std::vector<torch::Tensor> sdot_l2_forward(torch::Tensor f, torch::Tensor g) {
    auto result = torch::empty({1}, f.options());
    if (f.device().is_cpu()) {
        sdot_l2_cpu(f.data_ptr<float>(), g.data_ptr<float>(), f.numel(), result.data_ptr<float>());
    } else {
        auto f_cpu = f.to(torch::kCPU).contiguous();
        auto g_cpu = g.to(torch::kCPU).contiguous();
        float res_val;
        sdot_l2_cpu(f_cpu.data_ptr<float>(), g_cpu.data_ptr<float>(), f.numel(), &res_val);
        result.fill_(res_val);
    }
    return {result};
}

std::vector<torch::Tensor> sdot_l2_backward(torch::Tensor f, torch::Tensor g) {
    auto grad_f = torch::empty_like(f);
    auto grad_g = torch::empty_like(g);
    if (f.device().is_cpu()) {
        sdot_l2_backward_cpu(f.data_ptr<float>(), g.data_ptr<float>(), f.numel(), grad_f.data_ptr<float>(), grad_g.data_ptr<float>());
    } else {
        auto f_cpu = f.to(torch::kCPU).contiguous();
        auto g_cpu = g.to(torch::kCPU).contiguous();
        auto gf_cpu = torch::empty_like(f_cpu);
        auto gg_cpu = torch::empty_like(g_cpu);
        sdot_l2_backward_cpu(f_cpu.data_ptr<float>(), g_cpu.data_ptr<float>(), f.numel(), gf_cpu.data_ptr<float>(), gg_cpu.data_ptr<float>());
        grad_f.copy_(gf_cpu);
        grad_g.copy_(gg_cpu);
    }
    return {grad_f, grad_g};
}

std::vector<torch::Tensor> sdot_w2_forward(torch::Tensor Xf, torch::Tensor Wf, torch::Tensor Xg, torch::Tensor Yg) {
    int64_t batch_size = 1;
    int64_t Nf, Mg;
    
    if (Xf.dim() == 1) {
        batch_size = 1;
        Nf = Xf.size(0);
    } else {
        batch_size = Xf.size(0);
        Nf = Xf.size(1);
    }
    
    if (Xg.dim() == 1) {
        Mg = Xg.size(0);
        TORCH_CHECK(batch_size == 1, "Xg must have same batch size as Xf");
    } else {
        Mg = Xg.size(1);
        TORCH_CHECK(Xg.size(0) == batch_size, "Xg must have same batch size as Xf");
    }
    
    // Check Wf and Yg consistency
    TORCH_CHECK(Wf.sizes() == Xf.sizes(), "Wf and Xf must have same shapes");
    TORCH_CHECK(Yg.sizes() == Xg.sizes(), "Yg and Xg must have same shapes");
    
    auto result_shape = (Xf.dim() == 1) ? std::vector<int64_t>({1}) : std::vector<int64_t>({batch_size});
    auto result = torch::empty(result_shape, Xf.options());
    auto barycenters = torch::empty(Xf.sizes(), Xf.options());
    
    if (Xf.device().is_cpu()) {
        sdot_w2_cpu(Xf.data_ptr<float>(), Wf.data_ptr<float>(), Nf,
                    Xg.data_ptr<float>(), Yg.data_ptr<float>(), Mg,
                    batch_size,
                    result.data_ptr<float>(), barycenters.data_ptr<float>());
    } else {
        auto Xf_cpu = Xf.to(torch::kCPU).contiguous();
        auto Wf_cpu = Wf.to(torch::kCPU).contiguous();
        auto Xg_cpu = Xg.to(torch::kCPU).contiguous();
        auto Yg_cpu = Yg.to(torch::kCPU).contiguous();
        
        std::vector<float> res_vals(batch_size);
        std::vector<float> bary_vals(batch_size * Nf);
        
        sdot_w2_cpu(Xf_cpu.data_ptr<float>(), Wf_cpu.data_ptr<float>(), Nf,
                    Xg_cpu.data_ptr<float>(), Yg_cpu.data_ptr<float>(), Mg,
                    batch_size,
                    res_vals.data(), bary_vals.data());
        
        result.copy_(torch::from_blob(res_vals.data(), result_shape, torch::kFloat32));
        barycenters.copy_(torch::from_blob(bary_vals.data(), Xf.sizes(), torch::kFloat32));
    }
    
    return {result, barycenters};
}

std::vector<torch::Tensor> sdot_w2_backward(torch::Tensor grad_dist, torch::Tensor grad_bary, 
                                          torch::Tensor Xf, torch::Tensor Wf, torch::Tensor Xg, torch::Tensor Yg,
                                          torch::Tensor barycenters) {
    return {torch::zeros_like(Xf), torch::zeros_like(Wf), torch::zeros_like(Xg), torch::zeros_like(Yg)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdot_w2_forward, "SDOT W2 forward");
    m.def("backward", &sdot_w2_backward, "SDOT W2 backward");
    m.def("l2_forward", &sdot_l2_forward, "SDOT L2 forward");
    m.def("l2_backward", &sdot_l2_backward, "SDOT L2 backward");
}
