#include "Simulation.h"
#include "Config.h"
#include <iostream>

int main(int argc, char **argv) {
    Config cfg;
    // parse simple CLI args: first numeric arg is seed, --config-out or --config-out=path enables writing config
    for (int i=1;i<argc;++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "evo-movement-sim usage:\n";
            std::cout << "  --config-file=PATH            Load config overrides from PATH (JSON-like)\n";
            std::cout << "  --env-perlin-spatial-scale=N  Perlin spatial scale (double)\n";
            std::cout << "  --env-perlin-time-scale=N     Perlin temporal scale (double)\n";
            std::cout << "  --env-time-speed=N            Perlin time speed (double)\n";
            std::cout << "  --env-time-amplitude=N        Perlin time amplitude (double)\n";
            std::cout << "  --env-change-rate=N           Gaussian env noise sigma (double)\n";
            std::cout << "  --perlin-time / --no-perlin-time    Enable/disable perlin-time evolution\n";
            std::cout << "  --seed=N                      RNG seed (unsigned)\n";
            std::cout << "  --steps=N                     Number of timesteps to run (int)\n";
            std::cout << "  --report-interval=N           Reporting interval (int)\n";
            std::cout << "  --initial-population=N        Starting population size (int)\n";
            std::cout << "  --show-progress / --no-progress  Show textual progress bar and enable 'q' to stop\n";
            std::cout << "  --print-frames / --no-print-frames  Print ASCII frames to stdout\n";
            std::cout << "  --config-out=PATH             Write resolved config to PATH inside sim dir\n";
            std::cout << "  --help, -h                    Show this help message\n";
            return 0;
        }
        // runtime config file
        if (a.rfind("--config-file",0) == 0) {
            if (a.size() > 13 && a[13] == '=') {
                cfg.load_from_file(a.substr(14));
            } else if (a == "--config-file" && i+1 < argc) {
                cfg.load_from_file(argv[++i]);
            }
            continue;
        }
        // override flags
        if (a.rfind("--env-perlin-spatial-scale",0) == 0) {
            const std::string flag = "--env-perlin-spatial-scale";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.env_perlin_spatial_scale = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.env_perlin_spatial_scale = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--env-perlin-time-scale",0) == 0) {
            const std::string flag = "--env-perlin-time-scale";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.env_perlin_time_scale = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.env_perlin_time_scale = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--env-time-speed",0) == 0) {
            const std::string flag = "--env-time-speed";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.env_time_speed = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.env_time_speed = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--env-time-amplitude",0) == 0) {
            const std::string flag = "--env-time-amplitude";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.env_time_amplitude = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.env_time_amplitude = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--env-change-rate",0) == 0) {
            const std::string flag = "--env-change-rate";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.env_change_rate = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.env_change_rate = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--movement-cost-factor",0) == 0) {
            const std::string flag = "--movement-cost-factor";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.movement_cost_factor = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.movement_cost_factor = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--maintenance-cost-per-node",0) == 0) {
            const std::string flag = "--maintenance-cost-per-node";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.maintenance_cost_per_node = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.maintenance_cost_per_node = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a == "--no-perlin-time") { cfg.env_use_perlin_time = false; continue; }
        if (a == "--perlin-time") { cfg.env_use_perlin_time = true; continue; }
        if (a == "--no-progress") { cfg.show_progress = false; continue; }
        if (a == "--show-progress") { cfg.show_progress = true; continue; }
        if (a.rfind("--seed",0) == 0) {
            const std::string flag = "--seed";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.seed = std::stoul(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.seed = std::stoul(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--steps",0) == 0) {
            const std::string flag = "--steps";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.steps = std::stoi(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.steps = std::stoi(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--report-interval",0) == 0) {
            const std::string flag = "--report-interval";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.report_interval = std::stoi(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.report_interval = std::stoi(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--initial-population",0) == 0) {
            const std::string flag = "--initial-population";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.initial_population = std::stoi(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.initial_population = std::stoi(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--max-population",0) == 0) {
            const std::string flag = "--max-population";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.max_population = std::stoi(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.max_population = std::stoi(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--death-risk",0) == 0) {
            const std::string flag = "--death-risk";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.death_risk = std::stod(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) { try { cfg.death_risk = std::stod(argv[++i]); } catch(...) {} }
            continue;
        }
        if (a.rfind("--config-file",0) == 0) {
            if (a.size() > 13 && a[13] == '=') {
                cfg.load_from_file(a.substr(14));
            } else if (a == "--config-file" && i+1 < argc) {
                cfg.load_from_file(argv[++i]);
            }
            continue;
        }
        if (a.rfind("--config-out",0) == 0) {
            cfg.write_config = true;
            if (a.size() > 12 && a[12] == '=') {
                std::string p = a.substr(13);
                // If user passed an absolute path (Windows drive letter or leading slash),
                // sanitize to the basename so we don't record absolute paths inside the sim.
                bool looks_absolute = false;
                if (!p.empty() && (p[0] == '/' || p[0] == '\\')) looks_absolute = true;
                if (p.size() > 1 && p[1] == ':') looks_absolute = true; // e.g. Windows drive letter like 'C:' indicates an absolute path
                if (looks_absolute) {
                    size_t last = p.find_last_of("/\\");
                    if (last != std::string::npos) p = p.substr(last + 1);
                }
                cfg.config_out_path = p;
            }
        } else if (a.rfind("--burn-in",0) == 0) {
            const std::string flag = "--burn-in";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.burn_in = std::stoi(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) {
                try { cfg.burn_in = std::stoi(argv[++i]); } catch(...) {}
            }
        } else if (a.rfind("--steps",0) == 0) {
            const std::string flag = "--steps";
            if (a.size() > flag.size() && a[flag.size()] == '=') {
                try { cfg.steps = std::stoi(a.substr(flag.size()+1)); } catch(...) {}
            } else if (a == flag && i+1 < argc) {
                try { cfg.steps = std::stoi(argv[++i]); } catch(...) {}
            }
        } else if (a == "--print-frames") {
            cfg.print_frames = true;
        } else if (a == "--no-print-frames") {
            cfg.print_frames = false;
        } else {
            // try parse as seed
            try { cfg.seed = std::stoul(a); }
            catch(...) { /* ignore */ }
        }
    }

    Simulation sim(cfg);
    sim.init_population(cfg.initial_population);
    sim.run();
    
    return 0;
}
