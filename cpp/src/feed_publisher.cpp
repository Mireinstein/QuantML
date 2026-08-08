// UDP market-data publisher: sends synthetic tick messages over a real UDP
// socket. Loopback unicast to localhost, not multicast -- multicast group
// membership needs network configuration that isn't portable/testable in a
// generic dev sandbox, so this demonstrates the same socket-programming and
// wire-serialization skill on a configuration that reliably runs anywhere.
// A production feed would swap the unicast sendto() for a multicast group
// join, which is a socket-option change, not a different architecture.
#include "feed_protocol.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>

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
    const uint64_t n = argc > 2 ? std::stoul(argv[2]) : 100'000;

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "socket() failed\n";
        return 1;
    }

    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &dest.sin_addr);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> price_dist(9900, 10100);
    std::uniform_int_distribution<uint64_t> qty_dist(1, 100);
    std::bernoulli_distribution side_dist(0.5);

    // Give the subscriber a moment to bind before we start sending.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    for (uint64_t i = 0; i < n; ++i) {
        TickMessage msg{i, price_dist(rng), qty_dist(rng), static_cast<uint8_t>(side_dist(rng) ? 0 : 1), now_ns()};
        sendto(sock, &msg, sizeof(msg), 0, reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
    }

    std::cout << "Publisher: sent " << n << " ticks to 127.0.0.1:" << port << "\n";
    close(sock);
    return 0;
}
