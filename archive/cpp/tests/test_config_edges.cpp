#include "Config.h"
#include <iostream>
#include <fstream>
#include <filesystem>

int run_case_missing_file() {
    Config cfg;
    bool loaded = cfg.load_from_file("nonexistent_config_hopefully.json");
    if (loaded) { std::cerr << "Expected missing file to return false\n"; return 1; }
    return 0;
}

int run_case_malformed() {
    std::filesystem::path p("build/tests_config_malformed.json");
    std::ofstream out(p.string());
    out << "{ this is not : valid json ]";
    out.close();
    Config cfg;
    bool loaded = cfg.load_from_file(p.string());
    // loader is permissive; it may return true but should not crash; we accept either result
    cfg.validate();
    if (std::filesystem::exists(p)) std::filesystem::remove(p);
    return 0;
}

int run_case_negative_values() {
    std::filesystem::path p("build/tests_config_negative.json");
    std::ofstream out(p.string());
    out << "{\n\"steps\": -10, \"env_change_rate\": -0.5, \"env_perlin_spatial_scale\": -1.0\n}";
    out.close();
    Config cfg;
    cfg.load_from_file(p.string());
    cfg.validate();
    if (cfg.steps < 1) { std::cerr << "steps not clamped\n"; return 1; }
    if (cfg.env_change_rate < 0.0) { std::cerr << "env_change_rate not clamped\n"; return 1; }
    if (cfg.env_perlin_spatial_scale <= 0.0) { std::cerr << "env_perlin_spatial_scale not clamped\n"; return 1; }
    if (std::filesystem::exists(p)) std::filesystem::remove(p);
    return 0;
}

int run_case_boolean_edge() {
    std::filesystem::path p("build/tests_config_bool.json");
    std::ofstream out(p.string());
    out << "{\n\"env_use_perlin_time\": false, \"show_progress\": true\n}";
    out.close();
    Config cfg;
    cfg.load_from_file(p.string());
    cfg.validate();
    if (cfg.env_use_perlin_time != false) { std::cerr << "bool parse failed\n"; return 1; }
    if (cfg.show_progress != true) { std::cerr << "bool parse failed\n"; return 1; }
    if (std::filesystem::exists(p)) std::filesystem::remove(p);
    return 0;
}

int main() {
    if (run_case_missing_file()) return 1;
    if (run_case_malformed()) return 1;
    if (run_case_negative_values()) return 1;
    if (run_case_boolean_edge()) return 1;
    return 0;
}
