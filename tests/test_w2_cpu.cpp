#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "sdot_w2.h"

int main() {
    // Test 1: Single Dirac at 0.5, g(x) = 1 on [0,1]
    float Xf[] = {0.5f};
    float Wf[] = {1.0f};
    float Xg[] = {0.0f, 1.0f};
    float Yg[] = {1.0f, 1.0f};
    float result = -1.0f;
    float barycenters[1];
    
    sdot_w2_cpu(Xf, Wf, 1, Xg, Yg, 2, 1, &result, barycenters);
    std::cout << "Test 1 result: " << result << ", Barycenter: " << barycenters[0] << std::endl;
    assert(std::abs(result) < 1e-5);
    assert(std::abs(barycenters[0] - 0.5f) < 1e-5);
    
    // Test 2: Two Diracs at 0.25 and 0.75
    float Xf2[] = {0.25f, 0.75f};
    float Wf2[] = {0.5f, 0.5f};
    float barycenters2[2];
    sdot_w2_cpu(Xf2, Wf2, 2, Xg, Yg, 2, 1, &result, barycenters2);
    std::cout << "Test 2 result: " << result << ", Barycenters: " << barycenters2[0] << " " << barycenters2[1] << std::endl;
    assert(std::abs(result) < 1e-5);
    
    // Test 3: Batch of 2
    float Xf_batch[] = {0.5f, 0.0f}; // Test 1 and Test 3 from before
    float Wf_batch[] = {1.0f, 1.0f};
    float Xg_batch[] = {0.0f, 1.0f, 0.0f, 1.0f};
    float Yg_batch[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float results_batch[2];
    float bary_batch[2];
    
    sdot_w2_cpu(Xf_batch, Wf_batch, 1, Xg_batch, Yg_batch, 2, 2, results_batch, bary_batch);
    std::cout << "Batch result 0: " << results_batch[0] << " (exp 0.0)" << std::endl;
    std::cout << "Batch result 1: " << results_batch[1] << " (exp 0.25)" << std::endl;
    assert(std::abs(results_batch[0]) < 1e-5);
    assert(std::abs(results_batch[1] - 0.25f) < 1e-5);

    std::cout << "All Batched CPU W2 tests passed!" << std::endl;

    return 0;
}
