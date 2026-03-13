#include <mdspan>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> data(6, 1.0f);
    std::mdspan m(data.data(), 2, 3);
    // Tentative d'utiliser submdspan ou une alternative
    auto r0 = std::submdspan(m, 0, std::full_extent);
    std::cout << "Row 0 size: " << r0.extent(0) << std::endl;
    return 0;
}
