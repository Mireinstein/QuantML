// pybind11 bindings: exposes the real C++ OrderBook to Python so the
// backtester drives actual matching-engine fills instead of a separate
// simulated fill model.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "order_book.hpp"

namespace py = pybind11;
using namespace quantiq;

PYBIND11_MODULE(quantiq_cpp, m) {
    m.doc() = "Python bindings for the QuantIQ C++ limit order book";

    py::enum_<Side>(m, "Side")
        .value("Buy", Side::Buy)
        .value("Sell", Side::Sell);

    py::class_<Order>(m, "Order")
        .def(py::init([](uint64_t id, Side side, int64_t price, uint64_t qty, uint64_t ts) {
                 return Order{id, side, price, qty, ts};
             }),
             py::arg("id"), py::arg("side"), py::arg("price"), py::arg("qty"),
             py::arg("timestamp_ns") = 0)
        .def_readwrite("id", &Order::id)
        .def_readwrite("side", &Order::side)
        .def_readwrite("price", &Order::price)
        .def_readwrite("qty", &Order::qty)
        .def_readwrite("timestamp_ns", &Order::timestamp_ns);

    py::class_<Fill>(m, "Fill")
        .def_readonly("resting_order_id", &Fill::resting_order_id)
        .def_readonly("incoming_order_id", &Fill::incoming_order_id)
        .def_readonly("price", &Fill::price)
        .def_readonly("qty", &Fill::qty);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("submit", &OrderBook::submit, py::arg("order"),
             "Submit an order; returns the list of Fills it generated.")
        .def("cancel", &OrderBook::cancel, py::arg("order_id"))
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("bid_levels", &OrderBook::bid_levels)
        .def("ask_levels", &OrderBook::ask_levels);
}
