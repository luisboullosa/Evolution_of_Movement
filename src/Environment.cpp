#include "Environment.h"
#include "Perlin.h"
#include <algorithm>
#include <random>

void Environment::init_perlin(unsigned seed) {
    perlin_base = Perlin::generate(width, seed);
    resources = perlin_base;
}

void Environment::set_perlin_base(const std::vector<double> &base) {
    perlin_base = base;
    resources = perlin_base;
}

double Environment::gather_at(size_t pos, int local_population) {
    pos = pos % width;
    double available = resources[pos];
    double taken = 0.0;
    if (available <= 0.0) return 0.0;
    if (local_population <= 0) {
        taken = available;
    } else {
        taken = available / local_population;
    }
    resources[pos] = std::max(0.0, available - taken);
    return taken;
}

void Environment::step(double change_rate, std::mt19937 &rng, bool use_perlin_time, double time_speed, double time_amplitude, double spatial_scale, double time_scale) {
    if (use_perlin_time && !perlin_base.empty()) {
        perlin_time += time_speed;
        auto slice = Perlin::sample_time_slice(width, /*seed*/ 0, perlin_time, spatial_scale, time_scale);
        // combine base and slice scaled by amplitude
        for (size_t i = 0; i < resources.size(); ++i) {
            double v = perlin_base[i] + (slice[i] - 0.5) * 2.0 * time_amplitude; // normalize slice to [-1,1]
            // keep non-negative
            resources[i] = std::max(0.0, v);
        }
        return;
    }
    if (change_rate <= 0.0) return;
    std::normal_distribution<double> nd(0.0, change_rate);
    for (size_t i = 0; i < resources.size(); ++i) {
        double delta = nd(rng);
        resources[i] = std::max(0.0, resources[i] + delta);
    }
}
