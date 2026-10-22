#include "Simulation.h"
#include "Config.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

int main() {
    Config cfg;
    cfg.steps = 2;
    cfg.report_interval = 1;
    cfg.burn_in = 0; // ensure we write frames immediately
    cfg.write_frames = true;
    cfg.write_config = true;
    cfg.sim_prefix = "test";
    cfg.webviewer_base = "build/tests_out";
    cfg.sim_number = 0;
    cfg.frames_filename = "frames.tsv";
    cfg.env_out_path = "env.tsv";
    cfg.individuals_out_path = "individuals.tsv";
    cfg.config_out_path = "config_print.txt";

    // ensure clean start
    std::filesystem::path base(cfg.webviewer_base);
    std::filesystem::path simdir = base / "test0000";
    if (std::filesystem::exists(simdir)) std::filesystem::remove_all(simdir);
    std::filesystem::create_directories(base);

    Simulation sim(cfg);
    sim.init_population(3);
    sim.run();

    std::filesystem::path framep = simdir / cfg.frames_filename;
    if (!std::filesystem::exists(framep)) { std::cerr << "frames file missing\n"; return 1; }
    std::ifstream f(framep.string());
    std::string firstline;
    if (!std::getline(f, firstline)) { std::cerr << "frames file empty\n"; return 1; }
    if (firstline.rfind("#CONFIG", 0) != 0) { std::cerr << "frames header missing CONFIG: "<< firstline <<"\n"; f.close(); return 1; }
    f.close();

    std::filesystem::path cfgout = simdir / cfg.config_out_path;
    if (!std::filesystem::exists(cfgout)) { std::cerr << "config_out file missing\n"; return 1; }
    std::ifstream cf(cfgout.string());
    std::string cfgline;
    if (!std::getline(cf, cfgline)) { std::cerr << "config_out empty\n"; cf.close(); return 1; }
    // Accept either a '#CONFIG <json>' header (old behavior) or a plain JSON line (new behavior)
    if (cfgline.rfind("#CONFIG", 0) == 0) {
        // ok: header present
    } else if (!cfgline.empty() && cfgline[0] == '{') {
        // also acceptable: direct JSON content
    } else {
        std::cerr << "config_out content unexpected: "<<cfgline<<"\n"; cf.close(); return 1;
    }
    cf.close();

    // cleanup (give OS a moment to release file handles)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    try {
        if (std::filesystem::exists(simdir)) std::filesystem::remove_all(simdir);
        if (std::filesystem::exists(base) && std::filesystem::is_empty(base)) std::filesystem::remove(base);
    } catch (...) {
        // if cleanup fails on Windows due to locking, ignore in test
    }

    return 0;
}
