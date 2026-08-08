#pragma once

#include <cstddef>

namespace quantiq {

// Computes rolling (windowed) standard deviation of `returns` into `out`
// (both length n). out[i] is NaN for i+1 < window (insufficient history).
// Scalar reference implementation.
void rolling_volatility_scalar(const float* returns, size_t n, size_t window, float* out);

// NEON-vectorized implementation on ARM (falls back to the scalar path on
// platforms without NEON, so it's correct everywhere but only actually
// vectorized on ARM). Numerically equivalent to the scalar version --
// verified by tests -- and meant to be faster for large n/window.
void rolling_volatility_neon(const float* returns, size_t n, size_t window, float* out);

}  // namespace quantiq
