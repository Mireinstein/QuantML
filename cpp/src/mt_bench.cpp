// Multithreaded benchmark: a producer thread generates orders and pushes
// them through the lock-free SPSC ring buffer; a consumer thread pops them
// and submits to the real OrderBook. Measures the end-to-end latency from
// "order created" to "matching engine processed it" across a real thread
// handoff, not just single-threaded submit() cost (see main.cpp for that).
#include "order_book.hpp"
#include "ring_buffer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace quantiq;
using Clock = std::chrono::steady_clock;

namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

}  // namespace

int main(int argc, char** argv) {
    const size_t N = argc > 1 ? std::stoul(argv[1]) : 1'000'000;

    SpscRingBuffer<Order, 1 << 16> ring;
    std::atomic<bool> producer_done{false};

    std::thread producer([&] {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int64_t> price_dist(9900, 10100);
        std::uniform_int_distribution<uint64_t> qty_dist(1, 100);
        std::bernoulli_distribution side_dist(0.5);

        for (uint64_t i = 1; i <= N; ++i) {
            Order o{i, side_dist(rng) ? Side::Buy : Side::Sell, price_dist(rng), qty_dist(rng), now_ns()};
            while (!ring.push(o)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    OrderBook book;
    std::vector<uint64_t> handoff_latencies_ns;
    handoff_latencies_ns.reserve(N);

    std::thread consumer([&] {
        size_t processed = 0;
        while (processed < N) {
            auto order = ring.pop();
            if (order) {
                uint64_t latency = now_ns() - order->timestamp_ns;
                book.submit(*order);
                handoff_latencies_ns.push_back(latency);
                ++processed;
            } else if (!producer_done.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    });

    auto start = Clock::now();
    producer.join();
    consumer.join();
    auto end = Clock::now();

    double wall_seconds = std::chrono::duration<double>(end - start).count();

    std::sort(handoff_latencies_ns.begin(), handoff_latencies_ns.end());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * static_cast<double>(handoff_latencies_ns.size() - 1));
        return handoff_latencies_ns[idx];
    };

    std::cout << "Orders processed (producer thread -> ring buffer -> consumer thread -> OrderBook): " << N
              << "\n";
    std::cout << "Wall time: " << wall_seconds << "s (" << static_cast<uint64_t>(N / wall_seconds)
              << " orders/sec end-to-end across threads)\n";
    std::cout << "Final book depth: " << book.bid_levels() << " bid levels, " << book.ask_levels()
              << " ask levels\n";
    std::cout << "Producer->consumer handoff latency (ns), includes queueing + thread wake:\n";
    std::cout << "  p50:   " << pct(0.50) << "\n";
    std::cout << "  p90:   " << pct(0.90) << "\n";
    std::cout << "  p99:   " << pct(0.99) << "\n";
    std::cout << "  p99.9: " << pct(0.999) << "\n";
    std::cout << "  max:   " << handoff_latencies_ns.back() << "\n";

    return 0;
}
