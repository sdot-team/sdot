#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../src/cpu/sdot_l2_cpu.h"
#include <vector>

TEST_CASE("SDOT L2 CPU Forward", "[cpu][l2]") {
    const size_t N = 3;
    float f[] = {1.0f, 2.0f, 3.0f};
    float g[] = {1.0f, 1.0f, 1.0f};
    float result = 0.0f;
    
    sdot_l2_cpu(f, g, N, &result);
    
    float expected_result = (0.0f*0.0f + 1.0f*1.0f + 2.0f*2.0f) / 3.0f;
    REQUIRE_THAT(result, Catch::Matchers::WithinAbs(expected_result, 1e-5));
}

TEST_CASE("SDOT L2 CPU Backward", "[cpu][l2]") {
    const size_t N = 3;
    float f[] = {1.0f, 2.0f, 3.0f};
    float g[] = {1.0f, 1.0f, 1.0f};
    float grad_f[N];
    float grad_g[N];
    
    sdot_l2_backward_cpu(f, g, N, grad_f, grad_g);
    
    float expected_grad_f[] = {0.0f, 2.0f/3.0f, 4.0f/3.0f};
    float expected_grad_g[] = {0.0f, -2.0f/3.0f, -4.0f/3.0f};
    
    for (size_t i = 0; i < N; ++i) {
        CHECK_THAT(grad_f[i], Catch::Matchers::WithinAbs(expected_grad_f[i], 1e-5));
        CHECK_THAT(grad_g[i], Catch::Matchers::WithinAbs(expected_grad_g[i], 1e-5));
    }
}
