#include "../include/Config.h"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// Very small, permissive JSON-ish loader: looks for "key": value pairs where value may be
// number, boolean, or string (quoted). Not a full JSON parser but sufficient for small config files.
bool Config::load_from_file(const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto extract_number = [&](const std::string &k, double &out)->bool{
        size_t pos = content.find('"' + k + '"');
        if (pos == std::string::npos) return false;
        pos = content.find(':', pos);
        if (pos == std::string::npos) return false;
        size_t start = pos+1;
        while (start < content.size() && isspace((unsigned char)content[start])) start++;
        size_t end = start;
        bool has_sign = false;
        while (end < content.size() && (isdigit((unsigned char)content[end]) || content[end]=='.' || content[end]=='e' || content[end]=='E' || content[end]=='+' || content[end]=='-')) end++;
        if (end==start) return false;
        std::string tok = content.substr(start, end-start);
        try { out = std::stod(tok); return true; } catch(...) { return false; }
    };

    auto extract_int = [&](const std::string &k, int &out)->bool{
        double v; if (!extract_number(k, v)) return false; out = (int)v; return true;
    };

    auto extract_bool = [&](const std::string &k, bool &out)->bool{
        size_t pos = content.find('"' + k + '"');
        if (pos == std::string::npos) return false;
        pos = content.find(':', pos);
        if (pos == std::string::npos) return false;
        size_t start = pos+1; while (start < content.size() && isspace((unsigned char)content[start])) start++;
        if (content.compare(start, 4, "true") == 0) { out = true; return true; }
        if (content.compare(start, 5, "false") == 0) { out = false; return true; }
        return false;
    };

    auto extract_string = [&](const std::string &k, std::string &out)->bool{
        size_t pos = content.find('"' + k + '"');
        if (pos == std::string::npos) return false;
        pos = content.find(':', pos);
        if (pos == std::string::npos) return false;
        size_t start = content.find('"', pos+1);
        if (start == std::string::npos) return false;
        size_t end = content.find('"', start+1);
        if (end == std::string::npos) return false;
        out = content.substr(start+1, end-start-1);
        return true;
    };

    double dv;
    int iv;
    bool bv;
    std::string sv;

    if (extract_number("env_change_rate", dv)) env_change_rate = dv;
    if (extract_number("env_time_speed", dv)) env_time_speed = dv;
    if (extract_number("env_time_amplitude", dv)) env_time_amplitude = dv;
    if (extract_number("env_perlin_spatial_scale", dv)) env_perlin_spatial_scale = dv;
    if (extract_number("env_perlin_time_scale", dv)) env_perlin_time_scale = dv;
    if (extract_number("env_perlin_spatial_scale", dv)) env_perlin_spatial_scale = dv;
    if (extract_int("steps", iv)) steps = iv;
    if (extract_int("report_interval", iv)) report_interval = iv;
    if (extract_int("burn_in", iv)) burn_in = iv;
    if (extract_int("initial_population", iv)) initial_population = iv;
    if (extract_bool("env_use_perlin_time", bv)) env_use_perlin_time = bv;
    if (extract_bool("show_progress", bv)) show_progress = bv;
    if (extract_string("webviewer_base", sv)) webviewer_base = sv;
    if (extract_string("frames_filename", sv)) frames_filename = sv;

    return true;
}

bool Config::validate() {
    // Clamp/validate numeric ranges
    if (grid_width == 0) grid_width = 1;
    if (steps < 1) steps = 1;
    if (report_interval < 1) report_interval = 1;
    if (burn_in < 0) burn_in = 0;
    if (initial_population < 0) initial_population = 0;
    if (env_change_rate < 0.0) env_change_rate = 0.0;
    if (env_time_speed < 0.0) env_time_speed = 0.0;
    if (env_time_amplitude < 0.0) env_time_amplitude = 0.0;
    if (env_perlin_spatial_scale <= 0.0) env_perlin_spatial_scale = 0.05;
    if (env_perlin_time_scale <= 0.0) env_perlin_time_scale = 0.05;
    // tolerable defaults
    return true;
}
