#include "Simulation.h"
#include "Config.h"
#include <filesystem>
#include <iostream>
#include <fstream>

int main(){
    Config cfg;
    cfg.steps = 10;
    cfg.report_interval = 5;
    cfg.burn_in = 50; // larger than steps so nothing should be written
    cfg.write_frames = true;
    // write into a per-simulation folder under build/tests_out
    const std::string base = "build/tests_out";
    cfg.webviewer_base = base;
    cfg.sim_number = 0; // deterministic for the test -> build/tests_out/sim0000
    cfg.frames_filename = "frames.tsv";
    cfg.env_out_path = "env.tsv";
    cfg.individuals_out_path = "individuals.tsv";

    // ensure output dir exists and start from a clean sim directory
    std::filesystem::create_directories(base);
    std::filesystem::path simdir = std::filesystem::path(base) / "sim0000";
    if (std::filesystem::exists(simdir)) std::filesystem::remove_all(simdir);

    Simulation sim(cfg);
    sim.init_population(3);
    sim.run();

    // check that frames.tsv either does not exist or contains only header (no frame lines)
    std::filesystem::path f = simdir / cfg.frames_filename;
    if (std::filesystem::exists(f)) {
        // read file and check for lines after header
        std::ifstream in(f);
        std::string line; int lines = 0;
        while (std::getline(in, line)) lines++;
        // frames header has at least one line; if only header present then lines <= 1
        if (lines > 1) {
            std::cerr << "frames.tsv written despite burn-in; lines="<<lines<<"\n";
            // cleanup
            if (std::filesystem::exists(simdir)) std::filesystem::remove_all(simdir);
            return 1;
        }
    }
    // check that no env snapshot files exist for 0000 or 0005 inside the sim dir
    std::string env_fname = std::filesystem::path(cfg.env_out_path).filename().string();
    for (int t=0;t<=5;t+=5) {
        char buf[64]; sprintf(buf, "%04d", t);
        std::filesystem::path envp = simdir / (env_fname + std::string(".") + buf + std::string(".tsv"));
        if (std::filesystem::exists(envp)) {
            std::cerr << "Found snapshot "<< envp.string() <<" despite burn-in\n";
            if (std::filesystem::exists(simdir)) std::filesystem::remove_all(simdir);
            return 1;
        }
    }

    // cleanup test sim dir
    if (std::filesystem::exists(simdir)) std::filesystem::remove_all(simdir);
    // remove base if now empty
    if (std::filesystem::exists(base) && std::filesystem::is_empty(base)) std::filesystem::remove(base);

    return 0;
}
