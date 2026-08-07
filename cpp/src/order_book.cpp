#include "order_book.hpp"

#include <algorithm>

namespace quantiq {

std::vector<Fill> OrderBook::match_buy(Order& incoming) {
    std::vector<Fill> fills;
    while (incoming.qty > 0 && !asks_.empty()) {
        auto level_it = asks_.begin();
        if (level_it->first > incoming.price) break;  // no cross

        auto& level = level_it->second;
        while (incoming.qty > 0 && !level.empty()) {
            Order& resting = level.front();
            uint64_t traded = std::min(incoming.qty, resting.qty);
            fills.push_back(Fill{resting.id, incoming.id, level_it->first, traded});
            incoming.qty -= traded;
            resting.qty -= traded;
            if (resting.qty == 0) {
                locations_.erase(resting.id);
                level.pop_front();
            }
        }
        if (level.empty()) asks_.erase(level_it);
    }
    return fills;
}

std::vector<Fill> OrderBook::match_sell(Order& incoming) {
    std::vector<Fill> fills;
    while (incoming.qty > 0 && !bids_.empty()) {
        auto level_it = bids_.begin();
        if (level_it->first < incoming.price) break;  // no cross

        auto& level = level_it->second;
        while (incoming.qty > 0 && !level.empty()) {
            Order& resting = level.front();
            uint64_t traded = std::min(incoming.qty, resting.qty);
            fills.push_back(Fill{resting.id, incoming.id, level_it->first, traded});
            incoming.qty -= traded;
            resting.qty -= traded;
            if (resting.qty == 0) {
                locations_.erase(resting.id);
                level.pop_front();
            }
        }
        if (level.empty()) bids_.erase(level_it);
    }
    return fills;
}

void OrderBook::rest(const Order& order) {
    if (order.qty == 0) return;
    if (order.side == Side::Buy) {
        bids_[order.price].push_back(order);
    } else {
        asks_[order.price].push_back(order);
    }
    locations_[order.id] = Location{order.side, order.price};
}

std::vector<Fill> OrderBook::submit(Order order) {
    std::vector<Fill> fills = (order.side == Side::Buy) ? match_buy(order) : match_sell(order);
    rest(order);
    return fills;
}

bool OrderBook::cancel(uint64_t order_id) {
    auto it = locations_.find(order_id);
    if (it == locations_.end()) return false;

    const Side side = it->second.side;
    const int64_t price = it->second.price;
    auto erase_from = [&](auto& book) {
        auto level_it = book.find(price);
        if (level_it == book.end()) return;
        auto& dq = level_it->second;
        for (auto oit = dq.begin(); oit != dq.end(); ++oit) {
            if (oit->id == order_id) {
                dq.erase(oit);
                break;
            }
        }
        if (dq.empty()) book.erase(level_it);
    };

    if (side == Side::Buy) {
        erase_from(bids_);
    } else {
        erase_from(asks_);
    }
    locations_.erase(it);
    return true;
}

std::optional<int64_t> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<int64_t> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

}  // namespace quantiq
