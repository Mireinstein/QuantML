#pragma once

#include <cstdint>

namespace quantiq {

// Fixed-size wire format for the UDP market-data feed. Sent as raw bytes
// with no endianness conversion -- fine for same-machine loopback (this
// demo), but a real cross-host feed would need to fix the wire byte order
// (e.g. htonl/ntohl per field) since x86 and ARM disagree on it.
#pragma pack(push, 1)
struct TickMessage {
    uint64_t seq;
    int64_t price;
    uint64_t qty;
    uint8_t side;  // 0 = Buy, 1 = Sell
    uint64_t send_ts_ns;
};
#pragma pack(pop)

}  // namespace quantiq
