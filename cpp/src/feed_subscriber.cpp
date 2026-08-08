// UDP market-data subscriber: receives real tick messages over a UDP
// socket, submits each as an order to the real OrderBook, and reports
// end-to-end network+processing latency plus any packet loss (UDP has no
// delivery guarantee, even on loopback under load -- a real feed handler
// has to detect gaps via sequence numbers and this one does, though it
// doesn't implement gap recovery/snapshot resync, which is out of scope).
#include "feed_protocol.hpp"
#include "order_book.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    const uint16_t port = argc > 1 ? static_cast<uint16_t>(std::stoi(argv[1])) : 9001;
    const uint64_t expected_n = argc > 2 ? std::stoul(argv[2]) : 100'000;

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "socket() failed\n";
        return 1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind() failed on port " << port << "\n";
        return 1;
    }

    timeval tv{2, 0};  // 2s idle timeout so we exit even if packets were lost
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    OrderBook book;
    std::vector<uint64_t> latencies_ns;
    latencies_ns.reserve(expected_n);

    uint64_t received = 0;
    uint64_t max_seq_seen = 0;
    bool first = true;

    while (received < expected_n) {
        TickMessage msg;
        ssize_t n = recv(sock, &msg, sizeof(msg), 0);
        if (n != sizeof(msg)) {
            break;  // timeout or short read: publisher finished or packets were lost
        }
        uint64_t latency = now_ns() - msg.send_ts_ns;
        latencies_ns.push_back(latency);

        Order order{msg.seq, msg.side == 0 ? Side::Buy : Side::Sell, msg.price, msg.qty, msg.send_ts_ns};
        book.submit(order);

        if (first || msg.seq > max_seq_seen) {
            max_seq_seen = msg.seq;
            first = false;
        }
        ++received;
    }

    close(sock);

    std::cout << "Subscriber: received " << received << "/" << expected_n << " ticks ("
              << (expected_n - received) << " lost over UDP)\n";
    std::cout << "Final book depth: " << book.bid_levels() << " bid levels, " << book.ask_levels()
              << " ask levels\n";

    if (!latencies_ns.empty()) {
        std::sort(latencies_ns.begin(), latencies_ns.end());
        auto pct = [&](double p) {
            size_t idx = static_cast<size_t>(p * static_cast<double>(latencies_ns.size() - 1));
            return latencies_ns[idx];
        };
        std::cout << "Network + processing latency (ns), publisher send -> subscriber processed:\n";
        std::cout << "  p50: " << pct(0.50) << "\n";
        std::cout << "  p90: " << pct(0.90) << "\n";
        std::cout << "  p99: " << pct(0.99) << "\n";
        std::cout << "  max: " << latencies_ns.back() << "\n";
    }

    return 0;
}
