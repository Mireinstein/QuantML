#include "volatility.hpp"

#include <cmath>
#include <limits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace quantiq {

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
}

void rolling_volatility_scalar(const float* returns, size_t n, size_t window, float* out) {
    for (size_t i = 0; i < n; ++i) {
        if (i + 1 < window) {
            out[i] = kNaN;
            continue;
        }
        const float* w = returns + (i + 1 - window);
        float sum = 0.0f, sum_sq = 0.0f;
        for (size_t j = 0; j < window; ++j) {
            sum += w[j];
            sum_sq += w[j] * w[j];
        }
        float mean = sum / static_cast<float>(window);
        float variance = sum_sq / static_cast<float>(window) - mean * mean;
        out[i] = std::sqrt(variance < 0.0f ? 0.0f : variance);  // guard tiny negative fp error
    }
}

#if defined(__ARM_NEON)

namespace {
inline float hsum(float32x4_t v) {
    float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}
}  // namespace

void rolling_volatility_neon(const float* returns, size_t n, size_t window, float* out) {
    for (size_t i = 0; i < n; ++i) {
        if (i + 1 < window) {
            out[i] = kNaN;
            continue;
        }
        const float* w = returns + (i + 1 - window);

        float32x4_t sum_v = vdupq_n_f32(0.0f);
        float32x4_t sum_sq_v = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 4 <= window; j += 4) {
            float32x4_t x = vld1q_f32(w + j);
            sum_v = vaddq_f32(sum_v, x);
            sum_sq_v = vfmaq_f32(sum_sq_v, x, x);
        }
        float sum = hsum(sum_v);
        float sum_sq = hsum(sum_sq_v);
        for (; j < window; ++j) {  // remainder when window isn't a multiple of 4
            sum += w[j];
            sum_sq += w[j] * w[j];
        }

        float mean = sum / static_cast<float>(window);
        float variance = sum_sq / static_cast<float>(window) - mean * mean;
        out[i] = std::sqrt(variance < 0.0f ? 0.0f : variance);
    }
}

#else

void rolling_volatility_neon(const float* returns, size_t n, size_t window, float* out) {
    // No NEON on this platform: fall back to scalar so the function is
    // still correct everywhere, just not vectorized off of ARM.
    rolling_volatility_scalar(returns, n, window, out);
}

#endif

}  // namespace quantiq
