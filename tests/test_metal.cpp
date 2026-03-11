#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "sdot_l2.h"

int main() {
    const size_t N = 3;
    float f[] = {1.0f, 2.0f, 3.0f};
    float g[] = {1.0f, 1.0f, 1.0f};
    float result = 0.0f;
    
    sdot_l2_metal(f, g, N, &result);
    std::cout << "Metal Forward result: " << result << std::endl;
    
    float expected_result = (0.0f*0.0f + 1.0f*1.0f + 2.0f*2.0f) / 3.0f;
    assert(std::abs(result - expected_result) < 1e-5);
    std::cout << "Metal Forward check passed!" << std::endl;
    
    float grad_f[N];
    float grad_g[N];
    sdot_l2_backward_metal(f, g, N, grad_f, grad_g);
    
    std::cout << "Metal Grad F: ";
    for (size_t i = 0; i < N; ++i) std::cout << grad_f[i] << " ";
    std::cout << std::endl;
    
    float expected_grad_f[] = {0.0f, 2.0f/3.0f, 4.0f/3.0f};
    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(grad_f[i] - expected_grad_f[i]) < 1e-5);
    }
    std::cout << "Metal Backward F check passed!" << std::endl;

    std::cout << "Metal Grad G: ";
    for (size_t i = 0; i < N; ++i) std::cout << grad_g[i] << " ";
    std::cout << std::endl;

    float expected_grad_g[] = {0.0f, -2.0f/3.0f, -4.0f/3.0f};
    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(grad_g[i] - expected_grad_g[i]) < 1e-5);
    }
    std::cout << "Metal Backward G check passed!" << std::endl;

    return 0;
}
