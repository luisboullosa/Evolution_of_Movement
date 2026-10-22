#include "Perlin.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

static double interp(double a, double b, double t) {
    double ft = (1 - std::cos(t * M_PI)) * 0.5;
    return a*(1-ft) + b*ft;
}

std::vector<double> Perlin::generate(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> base(n);
    for (size_t i=0;i<n;++i) base[i] = dist(rng);

    // simple value-smooth noise using interpolation of nearby samples
    std::vector<double> out(n);
    for (size_t i=0;i<n;++i) {
        double x = (double)i * 0.1;
        int xi = (int)std::floor(x);
        double t = x - xi;
        double a = base[xi % n];
        double b = base[(xi+1) % n];
        out[i] = interp(a,b,t);
    }
    return out;
}

std::vector<double> Perlin::sample_time_slice(size_t n, unsigned seed, double time_offset, double spatial_scale, double time_scale) {
    // Proper 2D Perlin noise (space x time) implementation
    // Uses a seeded permutation table for deterministic gradients.
    auto fade = [](double t) {
        // classic fade: 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6 - 15) + 10);
    };
    auto lerp = [](double a, double b, double t) {
        return a + t * (b - a);
    };
    auto grad = [](int hash, double x, double y) {
        // 8 possible directions
        int h = hash & 7;
        double u = (h < 4) ? x : y;
        double v = (h < 4) ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0 * v : 2.0 * v) / 2.0;
    };

    // build permutation table
    std::vector<int> p(256);
    for (int i = 0; i < 256; ++i) p[i] = i;
    std::mt19937 rng(seed ? seed : 12345);
    std::shuffle(p.begin(), p.end(), rng);
    p.insert(p.end(), p.begin(), p.end()); // duplicate

    std::vector<double> out(n);
    // spatial_scale and time_scale are parameters passed in

    for (size_t i = 0; i < n; ++i) {
        double x = (double)i * spatial_scale;
        double y = time_offset * time_scale;

        int xi = (int)std::floor(x) & 255;
        int yi = (int)std::floor(y) & 255;
        double xf = x - std::floor(x);
        double yf = y - std::floor(y);

        double u = fade(xf);
        double v = fade(yf);

        int aa = p[p[xi] + yi];
        int ab = p[p[xi] + yi + 1];
        int ba = p[p[xi + 1] + yi];
        int bb = p[p[xi + 1] + yi + 1];

        double x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1.0, yf), u);
        double x2 = lerp(grad(ab, xf, yf - 1.0), grad(bb, xf - 1.0, yf - 1.0), u);
        double value = lerp(x1, x2, v);
        // value is roughly in range [-~1, ~1]; normalize to [0,1]
        out[i] = (value + 1.0) * 0.5;
    }
    return out;
}
