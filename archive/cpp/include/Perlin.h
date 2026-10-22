#pragma once
#include <vector>

struct Perlin {
    static std::vector<double> generate(size_t n, unsigned seed);
    // sample a Perlin-like time-offset slice for n points using the given seed.
    // time_offset can be fractional; the implementation will wrap/Interpolate.
    static std::vector<double> sample_time_slice(size_t n, unsigned seed, double time_offset, double spatial_scale = 0.05, double time_scale = 0.05);
};
