#pragma once

#include <cstdint>

namespace quantiq {

enum class Side : uint8_t { Buy, Sell };

struct Order {
    uint64_t id;
    Side side;
    int64_t price;      // integer ticks, avoids float rounding error
    uint64_t qty;
    uint64_t timestamp_ns;
};

struct Fill {
    uint64_t resting_order_id;
    uint64_t incoming_order_id;
    int64_t price;
    uint64_t qty;
};

}  // namespace quantiq
