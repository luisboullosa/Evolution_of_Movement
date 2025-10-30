#include "Simulation.h"
#include "Config.h"
#include <iostream>
#include <vector>

int main() {
    Config cfg;
    cfg.steps = 10;
    cfg.report_interval = 1; // ensure reporting happens each step
    cfg.burn_in = 3;        // randomization should occur at t == 3
    cfg.write_frames = false; // avoid file IO in this unit test
    cfg.seed = 42;          // deterministic RNG

    Simulation sim(cfg);
    const int npop = 5;
    sim.init_population(npop);

    // record positions before running
    std::vector<size_t> before;
    before.reserve(sim.individuals.size());
    for (const auto &ind : sim.individuals) before.push_back(ind.position);

    sim.run();

    // After burn-in randomization we expect at least one individual's position to differ
    bool any_changed = false;
    for (size_t i = 0; i < sim.individuals.size() && i < before.size(); ++i) {
        if (sim.individuals[i].position != before[i]) { any_changed = true; break; }
    }

    if (!any_changed) {
        std::cerr << "positions were not randomized after burn-in\n";
        return 1;
    }

    return 0;
}
