#include "Simulation.h"
#include "Config.h"
#include <filesystem>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    // prepare a clean output base under the build directory
    fs::path outbase = fs::current_path() / "test_sim_out";
    if (fs::exists(outbase)) fs::remove_all(outbase);
    fs::create_directories(outbase);

    Config cfg;
    cfg.webviewer_base = outbase.string();
    cfg.write_config = true;
    cfg.config_out_path = "config.json"; // request config to be written into sim dir
    cfg.steps = 1;
    cfg.initial_population = 1;
    cfg.write_frames = true;
    cfg.show_progress = false;

    Simulation sim(cfg);
    sim.init_population(cfg.initial_population);
    // run should write a simNNNN directory with config.json inside
    sim.run();

    // find any sim* directory inside outbase
    fs::path found;
    for (auto &p : fs::directory_iterator(outbase)) {
        if (!p.is_directory()) continue;
        std::string name = p.path().filename().string();
        if (name.rfind("sim", 0) == 0 || name.rfind("job", 0) == 0) { found = p.path(); break; }
    }
    if (found.empty()) {
        std::cerr << "No simulation directory created under " << outbase.string() << "\n";
        return 1;
    }

    fs::path cfgpath = found / "config.json";
    if (!fs::exists(cfgpath)) {
        std::cerr << "config.json not found in " << found.string() << "\n";
        return 2;
    }

    // try to load it back with Config::load_from_file
    Config loaded;
    bool ok = loaded.load_from_file(cfgpath.string());
    if (!ok) {
        std::cerr << "Failed to parse config.json at " << cfgpath.string() << "\n";
        return 3;
    }

    // basic checks
    if (loaded.steps != cfg.steps) {
        std::cerr << "Loaded steps mismatch: " << loaded.steps << " vs " << cfg.steps << "\n";
        return 4;
    }
    if (loaded.initial_population != cfg.initial_population) {
        std::cerr << "Loaded initial_population mismatch: " << loaded.initial_population << " vs " << cfg.initial_population << "\n";
        return 5;
    }

    // cleanup
    try { fs::remove_all(outbase); } catch(...) {}

    return 0;
}
