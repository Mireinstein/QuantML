#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

namespace quantiq {

// Lock-free single-producer single-consumer ring buffer. Capacity must be a
// power of two (enables index wraparound via masking instead of modulo).
//
// head_/tail_ are cache-line-aligned and separated so the producer writing
// head_ and the consumer writing tail_ don't false-share a cache line.
template <typename T, size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    // Producer-only. Returns false if the buffer is full.
    bool push(const T& value) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        buffer_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer-only. Returns std::nullopt if the buffer is empty.
    std::optional<T> pop() {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // empty
        }
        T value = buffer_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return value;
    }

    static constexpr size_t capacity() { return Capacity; }

private:
    static constexpr size_t mask_ = Capacity - 1;
    std::array<T, Capacity> buffer_{};

    alignas(64) std::atomic<size_t> head_{0};  // written by producer, read by consumer
    alignas(64) std::atomic<size_t> tail_{0};  // written by consumer, read by producer
};

}  // namespace quantiq
