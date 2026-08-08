#include "volatility.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace quantiq;

void test_neon_matches_scalar_on_random_data() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    for (size_t n : {10, 37, 100, 733}) {
        for (size_t window : {1, 4, 5, 20, 21}) {
            if (window > n) continue;
            std::vector<float> returns(n);
            for (auto& r : returns) r = dist(rng);

            std::vector<float> out_scalar(n), out_neon(n);
            rolling_volatility_scalar(returns.data(), n, window, out_scalar.data());
            rolling_volatility_neon(returns.data(), n, window, out_neon.data());

            for (size_t i = 0; i < n; ++i) {
                bool scalar_nan = std::isnan(out_scalar[i]);
                bool neon_nan = std::isnan(out_neon[i]);
                assert(scalar_nan == neon_nan);
                if (!scalar_nan) {
                    assert(std::fabs(out_scalar[i] - out_neon[i]) < 1e-4f);
                }
            }
        }
    }
}

void test_constant_series_has_zero_volatility() {
    std::vector<float> returns(50, 0.01f);
    std::vector<float> out(50);
    rolling_volatility_neon(returns.data(), 50, 10, out.data());
    for (size_t i = 9; i < 50; ++i) {
        assert(std::fabs(out[i]) < 1e-5f);
    }
}

void test_insufficient_window_is_nan() {
    std::vector<float> returns = {0.01f, 0.02f, -0.01f};
    std::vector<float> out(3);
    rolling_volatility_scalar(returns.data(), 3, 5, out.data());
    for (float v : out) assert(std::isnan(v));
}

int main() {
    test_neon_matches_scalar_on_random_data();
    test_constant_series_has_zero_volatility();
    test_insufficient_window_is_nan();
    std::cout << "All volatility tests passed.\n";
    return 0;
}
