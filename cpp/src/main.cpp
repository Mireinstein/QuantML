#include "order_book.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace quantiq;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const size_t N = argc > 1 ? std::stoul(argv[1]) : 1'000'000;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> price_dist(9900, 10100);  // ticks around a 10000 mid
    std::uniform_int_distribution<uint64_t> qty_dist(1, 100);
    std::bernoulli_distribution side_dist(0.5);

    OrderBook book;
    std::vector<uint64_t> latencies_ns;
    latencies_ns.reserve(N);

    for (uint64_t i = 1; i <= N; ++i) {
        Order o{i, side_dist(rng) ? Side::Buy : Side::Sell, price_dist(rng), qty_dist(rng), 0};
        auto start = Clock::now();
        book.submit(o);
        auto end = Clock::now();
        latencies_ns.push_back(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
    }

    std::sort(latencies_ns.begin(), latencies_ns.end());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * static_cast<double>(latencies_ns.size() - 1));
        return latencies_ns[idx];
    };

    std::cout << "Orders processed: " << N << "\n";
    std::cout << "Final book depth: " << book.bid_levels() << " bid levels, " << book.ask_levels()
              << " ask levels\n";
    std::cout << "Latency per submit() call (ns):\n";
    std::cout << "  p50:   " << pct(0.50) << "\n";
    std::cout << "  p90:   " << pct(0.90) << "\n";
    std::cout << "  p99:   " << pct(0.99) << "\n";
    std::cout << "  p99.9: " << pct(0.999) << "\n";
    std::cout << "  max:   " << latencies_ns.back() << "\n";

    return 0;
}
