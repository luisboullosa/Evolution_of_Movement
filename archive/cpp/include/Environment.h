#pragma once
#include <vector>
#include <random>

struct Environment {
    size_t width = 1000;
    std::vector<double> resources;
    // base perlin map used for time evolution
    std::vector<double> perlin_base;
    double perlin_time = 0.0;

    // Initialize perlin base and prepare resources
    void init_perlin(unsigned seed);
    // Initialize perlin base with an explicit base map (for testing/mocking)
    void set_perlin_base(const std::vector<double> &base);

    Environment() = default;
    double gather_at(size_t pos, int local_population);
    // advance environmental dynamics by one timestep (may apply small changes to resources)
    void step(double change_rate, std::mt19937 &rng, bool use_perlin_time = false, double time_speed = 0.0, double time_amplitude = 1.0, double spatial_scale = 0.05, double time_scale = 0.05);
};
