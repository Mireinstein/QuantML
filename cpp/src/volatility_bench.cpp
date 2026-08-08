// Benchmarks scalar vs NEON-vectorized rolling volatility to show the real
// speedup from vectorization on this machine, not a claimed one.
#include "volatility.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::stoul(argv[1]) : 2'000'000;
    const size_t window = argc > 2 ? std::stoul(argv[2]) : 20;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    std::vector<float> returns(n);
    for (auto& r : returns) r = dist(rng);

    std::vector<float> out(n);

    auto t0 = Clock::now();
    quantiq::rolling_volatility_scalar(returns.data(), n, window, out.data());
    auto t1 = Clock::now();
    quantiq::rolling_volatility_neon(returns.data(), n, window, out.data());
    auto t2 = Clock::now();

    double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double neon_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "n=" << n << " window=" << window << "\n";
    std::cout << "scalar: " << scalar_ms << " ms\n";
    std::cout << "neon:   " << neon_ms << " ms\n";
    std::cout << "speedup: " << (scalar_ms / neon_ms) << "x\n";
#if defined(__ARM_NEON)
    std::cout << "(NEON path active on this build)\n";
#else
    std::cout << "(No NEON on this platform -- neon path is the scalar fallback)\n";
#endif
    return 0;
}
