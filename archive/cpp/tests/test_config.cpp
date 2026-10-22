#include "Config.h"
#include <iostream>
#include <fstream>
#include <filesystem>

int main() {
    // create a temporary config file
    std::filesystem::path p("build/tests_config.json");
    std::ofstream out(p.string());
    out << "{\n";
    out << "  \"steps\": 42,\n";
    out << "  \"env_use_perlin_time\": true,\n";
    out << "  \"env_perlin_spatial_scale\": 0.02,\n";
    out << "  \"env_perlin_time_scale\": 0.03,\n";
    out << "  \"show_progress\": false,\n";
    out << "  \"webviewer_base\": \"build/test_viewer\"\n";
    out << "}\n";
    out.close();

    Config cfg;
    bool loaded = cfg.load_from_file(p.string());
    if (!loaded) { std::cerr << "Failed to load config file\n"; return 1; }
    cfg.validate();
    if (cfg.steps != 42) { std::cerr << "steps parse failed: "<<cfg.steps<<"\n"; return 1; }
    if (!cfg.env_use_perlin_time) { std::cerr << "env_use_perlin_time parse failed\n"; return 1; }
    if (std::abs(cfg.env_perlin_spatial_scale - 0.02) > 1e-9) { std::cerr << "spatial scale parse failed\n"; return 1; }
    if (std::abs(cfg.env_perlin_time_scale - 0.03) > 1e-9) { std::cerr << "time scale parse failed\n"; return 1; }
    if (cfg.show_progress != false) { std::cerr << "show_progress parse failed\n"; return 1; }
    if (cfg.webviewer_base != "build/test_viewer") { std::cerr << "webviewer_base parse failed\n"; return 1; }

    // cleanup
    if (std::filesystem::exists(p)) std::filesystem::remove(p);
    return 0;
}
