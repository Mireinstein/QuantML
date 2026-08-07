#pragma once

#include "types.hpp"

#include <deque>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace quantiq {

// Price-time priority limit order book. A resting order fills at its own
// (resting) price; an incoming order walks the opposite side while it
// crosses, then rests any unfilled remainder.
class OrderBook {
public:
    std::vector<Fill> submit(Order order);
    bool cancel(uint64_t order_id);

    std::optional<int64_t> best_bid() const;
    std::optional<int64_t> best_ask() const;

    size_t bid_levels() const { return bids_.size(); }
    size_t ask_levels() const { return asks_.size(); }

private:
    std::vector<Fill> match_buy(Order& incoming);
    std::vector<Fill> match_sell(Order& incoming);
    void rest(const Order& order);

    std::map<int64_t, std::deque<Order>, std::greater<int64_t>> bids_;  // best = highest price first
    std::map<int64_t, std::deque<Order>, std::less<int64_t>> asks_;     // best = lowest price first

    struct Location {
        Side side;
        int64_t price;
    };
    std::unordered_map<uint64_t, Location> locations_;
};

}  // namespace quantiq
