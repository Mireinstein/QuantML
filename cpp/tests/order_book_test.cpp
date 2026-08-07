#include "order_book.hpp"

#include <cassert>
#include <iostream>

using namespace quantiq;

int main() {
    // Resting order, no cross.
    {
        OrderBook book;
        auto fills = book.submit(Order{1, Side::Buy, 100, 10, 0});
        assert(fills.empty());
        assert(book.best_bid() == 100);
        assert(!book.best_ask().has_value());
    }

    // Crossing order, full fill, executes at the resting order's price.
    {
        OrderBook book;
        book.submit(Order{1, Side::Sell, 100, 10, 0});
        auto fills = book.submit(Order{2, Side::Buy, 105, 10, 0});
        assert(fills.size() == 1);
        assert(fills[0].price == 100);
        assert(fills[0].qty == 10);
        assert(!book.best_ask().has_value());
    }

    // Incoming larger than resting: partial fill, remainder rests.
    {
        OrderBook book;
        book.submit(Order{1, Side::Sell, 100, 5, 0});
        auto fills = book.submit(Order{2, Side::Buy, 100, 12, 0});
        assert(fills.size() == 1);
        assert(fills[0].qty == 5);
        assert(book.best_bid() == 100);
        assert(!book.best_ask().has_value());
    }

    // Cancel removes a resting order; double-cancel fails.
    {
        OrderBook book;
        book.submit(Order{1, Side::Buy, 100, 10, 0});
        assert(book.cancel(1));
        assert(!book.best_bid().has_value());
        assert(!book.cancel(1));
    }

    // Price-time priority: earlier resting order at the best price fills first.
    {
        OrderBook book;
        book.submit(Order{1, Side::Sell, 100, 5, 0});
        book.submit(Order{2, Side::Sell, 100, 5, 0});
        auto fills = book.submit(Order{3, Side::Buy, 100, 5, 0});
        assert(fills.size() == 1);
        assert(fills[0].resting_order_id == 1);
    }

    std::cout << "All order book tests passed.\n";
    return 0;
}
