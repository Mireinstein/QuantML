#include "ring_buffer.hpp"

#include <atomic>
#include <cassert>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

using namespace quantiq;

void test_fifo_order_single_threaded() {
    SpscRingBuffer<int, 8> rb;
    for (int i = 0; i < 5; ++i) {
        assert(rb.push(i));
    }
    for (int i = 0; i < 5; ++i) {
        auto v = rb.pop();
        assert(v.has_value());
        assert(*v == i);
    }
    assert(!rb.pop().has_value());
}

void test_full_when_capacity_minus_one_reached() {
    // One slot is always kept empty to distinguish full from empty.
    SpscRingBuffer<int, 4> rb;
    assert(rb.push(1));
    assert(rb.push(2));
    assert(rb.push(3));
    assert(!rb.push(4));  // full: capacity 4 holds at most 3 elements
}

void test_wraparound() {
    SpscRingBuffer<int, 4> rb;
    for (int round = 0; round < 100; ++round) {
        assert(rb.push(round));
        auto v = rb.pop();
        assert(v.has_value());
        assert(*v == round);
    }
}

void test_concurrent_producer_consumer_no_loss_no_corruption() {
    constexpr int kCount = 2'000'000;
    SpscRingBuffer<int, 4096> rb;
    std::atomic<bool> producer_done{false};

    std::thread producer([&] {
        for (int i = 0; i < kCount; ++i) {
            while (!rb.push(i)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    long long sum = 0;
    int received = 0;
    int expected_next = 0;
    std::thread consumer([&] {
        while (received < kCount) {
            auto v = rb.pop();
            if (v) {
                assert(*v == expected_next);  // SPSC ring buffer preserves order
                ++expected_next;
                sum += *v;
                ++received;
            } else if (producer_done.load(std::memory_order_acquire)) {
                // Drain any remaining items after producer finished.
                continue;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    long long expected_sum = 0;
    for (int i = 0; i < kCount; ++i) expected_sum += i;

    assert(received == kCount);
    assert(sum == expected_sum);
}

int main() {
    test_fifo_order_single_threaded();
    test_full_when_capacity_minus_one_reached();
    test_wraparound();
    test_concurrent_producer_consumer_no_loss_no_corruption();
    std::cout << "All ring buffer tests passed.\n";
    return 0;
}
