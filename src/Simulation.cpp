#include "Simulation.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <map>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

Simulation::Simulation(const Config &cfg) : rng(cfg.seed ? cfg.seed : 12345), config(cfg) {
    env.width = cfg.grid_width;
    env.init_perlin(cfg.seed ? cfg.seed : 12345);
    // set Individual static parameters from config
    Individual::maintenance_cost_per_node = cfg.maintenance_cost_per_node;
    Individual::reproduction_threshold = cfg.reproduction_threshold;
    Individual::reproduction_cost = cfg.reproduction_cost;
    Individual::mutation_rate = cfg.mutation_rate;
    Individual::mutation_sigma = cfg.mutation_sigma;
    Individual::movement_cost_factor = cfg.movement_cost_factor;
}

void Simulation::init_population(int n) {
    individuals.clear();
    std::uniform_int_distribution<size_t> pdist(0, env.width ? env.width - 1 : 0);
    for (int i=0;i<n;++i) {
        NeuralNetwork nn(config.nn_nodes, config.nn_inputs, config.nn_layers, config.nn_nodes2);
        // initialize weights and biases
        std::uniform_real_distribution<double> d(-1.0,1.0);
        if (nn.type == NeuralNetwork::Type::SINGLE_INPUT) {
            for (size_t wi = 0; wi < nn.weights.size(); ++wi) nn.weights[wi] = d(rng);
        } else {
            for (size_t wi = 0; wi < nn.weights_multi.size(); ++wi) nn.weights_multi[wi] = d(rng);
        }
        if (nn.layers == 2) {
            for (size_t wi = 0; wi < nn.weights2.size(); ++wi) nn.weights2[wi] = d(rng);
            for (size_t bi = 0; bi < nn.biases2.size(); ++bi) nn.biases2[bi] = d(rng);
        }
        for (size_t bi = 0; bi < nn.biases.size(); ++bi) nn.biases[bi] = d(rng);
        size_t pos = 0;
        if (config.randomize_start_positions) {
            pos = pdist(rng);
        } else {
            size_t w = env.width ? env.width : 1;
            pos = config.start_pos % w;
        }
        individuals.emplace_back(nn, pos, next_id++);
    }
}

void Simulation::step() {
    // advance environment dynamics
    try {
    env.step(config.env_change_rate, rng, config.env_use_perlin_time, config.env_time_speed, config.env_time_amplitude, config.env_perlin_spatial_scale, config.env_perlin_time_scale);
    } catch(...) {}
    // map positions to counts
    std::vector<int> counts(env.width, 0);
    for (auto &ind: individuals) counts[ind.position % env.width]++;

    last_spend.assign(individuals.size(), 0.0);
    last_delta.assign(individuals.size(), 0);
    // reset movers counter for this step (computed during action to avoid extra scan)
    last_movers = 0;

    // each individual acts
    struct MoveIntent { int idx; int newpos; };
    std::vector<MoveIntent> moves;

    for (size_t i=0;i<individuals.size();++i) {
        auto &ind = individuals[i];
        size_t pos = ind.position % env.width;
        double center = env.resources[pos];
        double left = env.resources[(pos + env.width - 1) % env.width];
        double right = env.resources[(pos + 1) % env.width];
        int occupancy = counts[pos];
    // pass detailed neighborhood inputs: left, center, right, occupancy
    double out = ind.act(left, center, right, occupancy, rng);
        if (out != 0.0) {
            // move: compute new position based on out (signed)
            int delta = (int)std::round(out);
            last_spend[i] = std::abs(out);
            last_delta[i] = delta;
            // count this mover (zero-cost increment)
            ++last_movers;
            int new_position = (int)ind.position + delta;
            while (new_position < 0) new_position += env.width;
            new_position = new_position % (int)env.width;
            ind.position = (size_t)new_position;
        }
    }

    // age individuals (increment age)
    for (auto &ind: individuals) ind.age += 1;

    // reproduction: simple rule: if resources > threshold -> reproduce
    size_t before = individuals.size();
    for (size_t i=0;i<before; ++i) {
        auto &ind = individuals[i];
        if (ind.resources > Individual::reproduction_threshold) {
            // respect optional max population cap
            if (config.max_population > 0 && (int)individuals.size() >= config.max_population) break;
            Individual child = ind.reproduce(rng, next_id);
            // reset child's age
            child.age = 0;
            individuals.push_back(child);
        }
    }

    // maintenance cost
    for (auto &ind: individuals) {
        size_t total_nodes = ind.nn.nodes + ind.nn.nodes2;
        ind.resources -= double(total_nodes) * Individual::maintenance_cost_per_node;
    }

    // remove dead individuals
    // apply age-related death risk per individual (probabilistic) and remove dead/reached-zero-resource
    std::uniform_real_distribution<double> ud(0.0,1.0);
    individuals.erase(std::remove_if(individuals.begin(), individuals.end(), [&](const Individual &ind){
        if (ind.resources <= 0.0) return true;
        if (config.death_risk > 0.0 && ind.age > 0) {
            // interprets death probability from risk * age in an exponential hazard style
            double p = 1.0 - std::exp(-config.death_risk * double(ind.age));
            double r = ud(rng);
            if (r < p) return true;
        }
        return false;
    }), individuals.end());
}

void Simulation::dump_env(const std::string &path) const {
    // Append a single NDJSON line to env.jsonl in the same directory as `path`.
    try {
        // prefer explicit sim target directory (set by run()), else derive from `path`
        std::string dir = nn_index_target_dir.empty() ? std::string(".") : nn_index_target_dir;
        if (dir == ".") {
            auto pos = path.find_last_of("/\\");
            if (pos != std::string::npos) dir = path.substr(0, pos);
        }
    // attempt to extract timestamp from filename (between the last two dots)
    size_t fpos = path.find_last_of("/\\");
    std::string fname = (fpos != std::string::npos) ? path.substr(fpos+1) : path;
    size_t lastdot = fname.find_last_of('.');
    size_t prevdot = std::string::npos;
        if (lastdot != std::string::npos) prevdot = fname.find_last_of('.', lastdot-1);
        int tval = -1;
        if (prevdot != std::string::npos && lastdot > prevdot + 1) {
            std::string tstr = fname.substr(prevdot+1, lastdot - (prevdot+1));
            try { tval = std::stoi(tstr); } catch(...) { tval = -1; }
        }
    std::filesystem::create_directories(dir);
    std::filesystem::path p(dir);
    p /= "env.jsonl";
        std::ofstream jf(p.string(), std::ofstream::out | std::ofstream::app);
        if (!jf.is_open()) return;
        jf << std::fixed << std::setprecision(6);
        jf << "{";
        if (tval >= 0) jf << "\"t\":" << tval << ", ";
        jf << "\"resources\": [";
        for (size_t i=0;i<env.width;++i) {
            if (i) jf << ", "; jf << env.resources[i];
        }
        jf << "]}" << "\n";
        jf.close();
    } catch(...) {
        // ignore write errors
    }
}

void Simulation::dump_individuals(const std::string &path) const {
    // Append a single NDJSON line to individuals.jsonl in the same directory as `path`.
    try {
        // prefer explicit sim target directory (set by run()), else derive from `path`
        std::string dir = nn_index_target_dir.empty() ? std::string(".") : nn_index_target_dir;
        if (dir == ".") {
            auto pos = path.find_last_of("/\\");
            if (pos != std::string::npos) dir = path.substr(0, pos);
        }
    // attempt to extract timestamp from filename (between the last two dots)
    size_t fpos = path.find_last_of("/\\");
    std::string fname = (fpos != std::string::npos) ? path.substr(fpos+1) : path;
    size_t lastdot = fname.find_last_of('.');
    size_t prevdot = std::string::npos;
        if (lastdot != std::string::npos) prevdot = fname.find_last_of('.', lastdot-1);
        int tval = -1;
        if (prevdot != std::string::npos && lastdot > prevdot + 1) {
            std::string tstr = fname.substr(prevdot+1, lastdot - (prevdot+1));
            try { tval = std::stoi(tstr); } catch(...) { tval = -1; }
        }
    std::filesystem::create_directories(dir);
    std::filesystem::path p(dir);
    p /= "individuals.jsonl";
        std::ofstream jf(p.string(), std::ofstream::out | std::ofstream::app);
        if (!jf.is_open()) return;
        jf << std::fixed << std::setprecision(6);
        jf << "{";
        if (tval >= 0) jf << "\"t\":" << tval << ", ";
        jf << "\"individuals\": [";
        for (size_t i=0;i<individuals.size();++i) {
            const auto &ind = individuals[i];
            double spend = (i < last_spend.size()) ? last_spend[i] : 0.0;
            int delta = (i < last_delta.size()) ? last_delta[i] : 0;
            if (i) jf << ", ";
            jf << "{";
            jf << "\"id\":" << ind.id << ", ";
            jf << "\"pos\":" << ind.position << ", ";
            jf << "\"resources\":" << ind.resources << ", ";
            jf << "\"last_spend\":" << spend << ", ";
            jf << "\"last_delta\":" << delta << ", ";
            jf << "\"age\":" << ind.age;
            jf << "}";
        }
        jf << "]}" << "\n";
        jf.close();
    } catch(...) {
        // ignore write errors
    }

    // Update in-memory nn_index cache instead of writing to disk repeatedly.
    try {
        // determine directory for index: prefer existing nn_index_target_dir if set
        std::string dir_for_index = nn_index_target_dir.empty() ? std::string(".") : nn_index_target_dir;
        if (dir_for_index == ".") {
            size_t fpos2 = path.find_last_of("/\\");
            if (fpos2 != std::string::npos) dir_for_index = path.substr(0, fpos2);
        }
        // Save target directory so run() can flush at the end
        nn_index_target_dir = dir_for_index;
        for (size_t i=0;i<individuals.size();++i) {
            const auto &ind = individuals[i];
            std::string idstr = std::to_string(ind.id);
            if (nn_index_cache.find(idstr) != nn_index_cache.end()) continue;
            std::ostringstream of;
            of << "{\n";
            // metadata
            of << "      \"type\": \"" << (ind.nn.type == NeuralNetwork::Type::SINGLE_INPUT ? "single" : "multi") << "\"," << "\n";
            of << "      \"inputs\": " << ind.nn.inputs << ",\n";
            of << "      \"nodes\": " << ind.nn.nodes << ",\n";
            of << "      \"layers\": " << ind.nn.layers << ",\n";
            of << "      \"nodes2\": " << ind.nn.nodes2 << ",\n";

            // weights
            of << "      \"weights\": ";
            if (ind.nn.type == NeuralNetwork::Type::SINGLE_INPUT) {
                of << "[";
                for (size_t wi=0; wi<ind.nn.weights.size(); ++wi) {
                    if (wi) of << ", "; of << std::fixed << std::setprecision(6) << ind.nn.weights[wi];
                }
                of << "]";
            } else {
                of << "[";
                for (size_t ni=0; ni<ind.nn.nodes;++ni) {
                    if (ni) of << ", ";
                    of << "[";
                    for (size_t j=0;j<ind.nn.inputs;++j) {
                        if (j) of << ", "; of << std::fixed << std::setprecision(6) << ind.nn.weights_multi[ni * ind.nn.inputs + j];
                    }
                    of << "]";
                }
                of << "]";
            }

            // biases
            of << ",\n      \"biases\": [";
            for (size_t bi=0; bi<ind.nn.biases.size(); ++bi) {
                if (bi) of << ", "; of << std::fixed << std::setprecision(6) << ind.nn.biases[bi];
            }
            of << "]";

            // optional second layer
            if (ind.nn.layers == 2) {
                of << ",\n      \"layer2_weights\": [";
                for (size_t ni=0; ni<ind.nn.nodes2; ++ni) {
                    if (ni) of << ", ";
                    of << "[";
                    for (size_t j=0; j<ind.nn.nodes; ++j) {
                        if (j) of << ", "; of << std::fixed << std::setprecision(6) << ind.nn.weights2[ni * ind.nn.nodes + j];
                    }
                    of << "]";
                }
                of << "],\n      \"layer2_biases\": [";
                for (size_t bi=0; bi<ind.nn.biases2.size(); ++bi) {
                    if (bi) of << ", "; of << std::fixed << std::setprecision(6) << ind.nn.biases2[bi];
                }
                of << "]";
            }

            of << "\n    }";
            nn_index_cache.emplace(std::to_string(ind.id), of.str());
            nn_index_dirty = true;
        }
    } catch (...) {
        // ignore cache update errors
    }
}

#include <fstream>

void Simulation::run() {
    int steps = config.steps;
    int report_interval = config.report_interval;
    bool write_frames = config.write_frames;
    std::atomic<bool> stop_flag(false);
    std::thread input_thread;
    if (config.show_progress) {
        // spawn a background thread that waits for user input; typing 'q' + Enter sets stop_flag
        input_thread = std::thread([&stop_flag]() {
            std::string s;
            while (std::getline(std::cin, s)) {
                if (!s.empty() && (s[0] == 'q' || s[0] == 'Q')) { stop_flag = true; break; }
            }
        });
    }
    const std::string frames_path = config.webviewer_base + std::string("/") + config.frames_filename;

    std::ofstream fout;
    std::string simdir;
    if (write_frames) {
        // try to create a per-simulation directory under webviewer_base
        try {
            std::filesystem::path basepath(config.webviewer_base);
            std::filesystem::create_directories(basepath);
            int simnum = config.sim_number;
            if (simnum < 0) {
                int maxn = -1;
                for (auto &p: std::filesystem::directory_iterator(basepath)) {
                    if (!p.is_directory()) continue;
                    std::string nm = p.path().filename().string();
                    if (nm.size() >= config.sim_prefix.size() && nm.rfind(config.sim_prefix,0)==0) {
                        std::string s = nm.substr(config.sim_prefix.size());
                        try { int v = std::stoi(s); if (v > maxn) maxn = v; } catch(...) {}
                    }
                }
                simnum = maxn + 1;
            }
            std::ostringstream dn;
            dn << config.sim_prefix << std::setw(config.sim_digits) << std::setfill('0') << simnum;
            std::filesystem::path simpath = basepath / dn.str();
            std::filesystem::create_directories(simpath);
            simdir = simpath.string();
            std::filesystem::path fpath = simpath / config.frames_filename;
            fout.open(fpath.string(), std::ofstream::out | std::ofstream::trunc);
            if (fout.is_open()) {
                // optionally write resolved config as a JSON-like header line
                if (config.write_config) {
                    // Build a machine-readable JSON representation of the resolved config.
                    std::ostringstream js;
                    js << "{";
                    auto put_kv = [&](const std::string &k, const std::string &v, bool last){ js << "\"" << k << "\":" << v; if (!last) js << ","; };
                    // numeric and boolean fields
                    put_kv("grid_width", std::to_string(config.grid_width), false);
                    put_kv("seed", std::to_string(config.seed), false);
                    put_kv("resource_scale", std::to_string(config.resource_scale), false);
                    put_kv("env_use_perlin_time", (config.env_use_perlin_time?"true":"false"), false);
                    put_kv("env_time_speed", std::to_string(config.env_time_speed), false);
                    put_kv("env_time_amplitude", std::to_string(config.env_time_amplitude), false);
                    put_kv("env_perlin_spatial_scale", std::to_string(config.env_perlin_spatial_scale), false);
                    put_kv("env_perlin_time_scale", std::to_string(config.env_perlin_time_scale), false);
                    put_kv("env_change_rate", std::to_string(config.env_change_rate), false);
                    put_kv("initial_population", std::to_string(config.initial_population), false);
                    put_kv("max_population", std::to_string(config.max_population), false);
                    put_kv("randomize_start_positions", (config.randomize_start_positions?"true":"false"), false);
                    put_kv("start_pos", std::to_string(config.start_pos), false);
                    put_kv("nn_nodes", std::to_string(config.nn_nodes), false);
                    put_kv("nn_layers", std::to_string(config.nn_layers), false);
                    put_kv("nn_nodes2", std::to_string(config.nn_nodes2), false);
                    put_kv("nn_inputs", std::to_string(config.nn_inputs), false);
                    put_kv("nn_use_occupancy", (config.nn_use_occupancy?"true":"false"), false);
                    put_kv("steps", std::to_string(config.steps), false);
                    put_kv("report_interval", std::to_string(config.report_interval), false);
                    put_kv("burn_in", std::to_string(config.burn_in), false);
                    put_kv("reproduction_threshold", std::to_string(config.reproduction_threshold), false);
                    put_kv("reproduction_cost", std::to_string(config.reproduction_cost), false);
                    put_kv("death_risk", std::to_string(config.death_risk), false);
                    put_kv("maintenance_cost_per_node", std::to_string(config.maintenance_cost_per_node), false);
                    put_kv("mutation_rate", std::to_string(config.mutation_rate), false);
                    put_kv("mutation_sigma", std::to_string(config.mutation_sigma), false);
                    put_kv("movement_cost_factor", std::to_string(config.movement_cost_factor), false);
                    put_kv("write_frames", (config.write_frames?"true":"false"), false);
                    // string fields (escape simple quotes)
                    auto esc = [&](const std::string &s){ std::string out; for(char c: s){ if (c=='\\') out += "\\\\"; else if (c=='\"') out += "\\\""; else out.push_back(c);} return out; };
                    put_kv("webviewer_base", std::string("\"") + esc(config.webviewer_base) + std::string("\""), false);
                    put_kv("sim_prefix", std::string("\"") + esc(config.sim_prefix) + std::string("\""), false);
                    put_kv("frames_filename", std::string("\"") + esc(config.frames_filename) + std::string("\""), false);
                    put_kv("env_out_path", std::string("\"") + esc(config.env_out_path) + std::string("\""), false);
                    put_kv("individuals_out_path", std::string("\"") + esc(config.individuals_out_path) + std::string("\""), false);
                    put_kv("config_out_path", std::string("\"") + esc(config.config_out_path) + std::string("\""), false);
                    put_kv("nn_index_flush_interval", std::to_string(config.nn_index_flush_interval), false);
                    put_kv("nn_index_flush_size", std::to_string(config.nn_index_flush_size), true);
                    js << "}";
                    std::string json_str = js.str();

                    // write a human-visible header into frames file (prefixed with #CONFIG)
                    std::ostringstream header;
                    header << "#CONFIG " << json_str << "\n";
                    fout << header.str();

                    // also write a machine-readable config.json inside the sim directory
                    try {
                        std::filesystem::path cfgpath = std::filesystem::path(simpath) / std::string("config.json");
                        std::ofstream cfgout(cfgpath.string(), std::ofstream::out | std::ofstream::trunc);
                        if (cfgout.is_open()) {
                            cfgout << json_str << std::endl;
                            cfgout.close();
                        }
                        // if a custom config_out_path was requested, write the same JSON there as well
                        if (!config.config_out_path.empty()) {
                            std::filesystem::path custom = std::filesystem::path(simpath) / config.config_out_path;
                            std::ofstream coutf(custom.string(), std::ofstream::out | std::ofstream::trunc);
                            if (coutf.is_open()) { coutf << json_str << std::endl; coutf.close(); }
                        }
                    } catch(...) {
                        // ignore write errors
                    }
                }
                fout << "t\tpop\tavg_res\tframe\tprop_moved\n";
            }
        } catch (...) {
            // fallback to legacy path
            fout.open(frames_path, std::ofstream::out | std::ofstream::trunc);
            if (fout.is_open()) {
                if (config.write_config) {
                    std::ostringstream ss;
                    ss << "#CONFIG {\"grid_width\":" << config.grid_width << "}\n";
                    fout << ss.str();
                }
                fout << "t\tpop\tavg_res\tframe\tprop_moved\n";
            }
        }
    }
    // record start time for ETA calculations when showing progress
    auto start_time = std::chrono::steady_clock::now();
    for (int t=0;t<steps && !stop_flag.load();++t) {
        step();
        // stop simulation if there are no individuals left
        if (individuals.empty()) {
            std::cout << "[info] population extinct at t=" << t << ", stopping simulation\n";
            break;
        }
        if (config.show_progress) {
            // simple textual progress bar to stdout with ETA and stop hint
            const int bar_width = 40;
            double frac = (double)(t+1) / (double)steps;
            int pos = (int)(bar_width * frac);
            std::ostringstream ss;
            ss << "[";
            for (int i=0;i<bar_width;++i) ss << (i<pos ? '=' : (i==pos ? '>' : ' '));
            ss << "] ";
            ss << int(frac * 100.0) << "%";

            // compute ETA based on elapsed time per completed step
            try {
                using namespace std::chrono;
                auto now = steady_clock::now();
                double elapsed = duration_cast<duration<double>>(now - start_time).count();
                double avg_per_step = elapsed / double(t+1);
                int remaining_steps = std::max(0, steps - (t+1));
                int eta_seconds = (int)std::round(avg_per_step * remaining_steps);
                int hrs = eta_seconds / 3600;
                int mins = (eta_seconds % 3600) / 60;
                int secs = eta_seconds % 60;
                ss << "  ETA " << hrs << ":" << (mins<10?"0":"") << mins << ":" << (secs<10?"0":"") << secs;
            } catch (...) {
                // ignore ETA errors
            }

            // append current timestep, population size and prop_moved (computed from last_movers to avoid extra scan)
            try {
                double prop_now = 0.0;
                size_t seen = std::min(last_delta.size(), individuals.size());
                if (seen > 0) prop_now = double(last_movers) / double(seen);
                ss << "  t=" << t << " pop=" << individuals.size() << " prop_moved=" << std::fixed << std::setprecision(3) << prop_now;
            } catch (...) {}

            ss << "  (ctrl+c to stop simulation)\r";
            std::cout << ss.str() << std::flush;
        }
        if (t % report_interval == 0) {
            if (t < config.burn_in) {
                // still run simulation steps but skip reporting until burn-in is over
                continue;
            }
            // when we've just reached the end of the burn-in period, randomize positions once
            if (t == config.burn_in) {
                try {
                    std::uniform_int_distribution<size_t> pdist(0, env.width ? env.width - 1 : 0);
                    for (auto &ind : individuals) {
                        ind.position = pdist(rng);
                    }
                    std::cout << "[info] randomized individual positions after burn-in\n";
                } catch (...) {
                    // ignore if randomization fails
                }
            }

            // When performing the geographic reset after burn-in, also reset individual resources
            // to their starter/default values so everyone begins the next phase equally.
            if (t == config.burn_in) {
                try {
                    for (auto &ind : individuals) {
                        ind.resources = Individual().resources; // default starter value (e.g., 10.0)
                    }
                } catch (...) {
                    // ignore reset failures
                }
            }
            double avgres = 0.0;
            for (auto &ind: individuals) avgres += ind.resources;
            if (!individuals.empty()) avgres /= individuals.size();
            // compute proportion of moving individuals for reporting (use last_movers computed during step)
            double prop_moved = 0.0;
            try {
                size_t seen = std::min(last_delta.size(), individuals.size());
                if (seen > 0) prop_moved = double(last_movers) / double(seen);
            } catch (...) { prop_moved = 0.0; }
            // std::cout << "t="<<t<<" pop="<<individuals.size()<<" avg_res="<<avgres<<"\n";
            std::string frame = get_visualization(config.vis_cols, config.vis_height);
            if (config.print_frames) std::cout << frame << "\n";
            if (fout.is_open()) {
                // escape newlines so TSV stays one-line per frame: replace '\n' with two-character sequence \ and n
                std::string esc;
                esc.reserve(frame.size()*2);
                for (char ch: frame) {
                    if (ch == '\n') { esc.push_back('\\'); esc.push_back('n'); }
                    else esc.push_back(ch);
                }
                // include proportion moved as an extra TSV column
                fout << t << '\t' << individuals.size() << '\t' << avgres << '\t' << esc << '\t' << prop_moved << '\n';
            }
            // also dump env and individuals snapshots for visualization (timestamped)
            char buf[64];
            sprintf(buf, "%04d", t);
            std::string envpath;
            std::string indpath;
            if (!simdir.empty()) {
                // use filename portion of configured paths when writing into simdir
                std::string env_fname = std::filesystem::path(config.env_out_path).filename().string();
                std::string ind_fname = std::filesystem::path(config.individuals_out_path).filename().string();
                envpath = (std::filesystem::path(simdir) / (env_fname + std::string(".") + buf + ".tsv")).string();
                indpath = (std::filesystem::path(simdir) / (ind_fname + std::string(".") + buf + ".tsv")).string();
            } else {
                envpath = config.env_out_path + "." + buf + ".tsv";
                indpath = config.individuals_out_path + "." + buf + ".tsv";
            }
            dump_env(envpath);
            dump_individuals(indpath);

            // Periodic flush: if configured, write nn_index.json when interval or size reached
            try {
                auto flush_if_needed = [&]() {
                    if (!nn_index_dirty) return;
                    bool do_flush = false;
                    if (config.nn_index_flush_interval > 0 && (t % config.nn_index_flush_interval) == 0) do_flush = true;
                    if (config.nn_index_flush_size > 0 && nn_index_cache.size() >= config.nn_index_flush_size) do_flush = true;
                    if (!do_flush) return;
                    std::filesystem::path p(nn_index_target_dir.empty() ? config.webviewer_base : nn_index_target_dir);
                    std::filesystem::create_directories(p);
                    p /= "nn_index.json";
                    std::ofstream jf(p.string(), std::ofstream::out | std::ofstream::trunc);
                    if (jf.is_open()) {
                        jf << "{\n";
                        size_t cnt = 0;
                        for (const auto &kv: nn_index_cache) {
                            jf << "  \"" << kv.first << "\": " << kv.second;
                            ++cnt;
                            if (cnt < nn_index_cache.size()) jf << ",\n"; else jf << "\n";
                        }
                        jf << "}\n";
                        jf.close();
                    }
                    nn_index_dirty = false;
                };
                flush_if_needed();
            } catch(...) {
                // ignore flush errors
            }
        }
    }
    if (fout.is_open()) fout.close();
    if (config.show_progress) {
        std::cout << std::endl;
        if (input_thread.joinable()) {
            // try to close input thread by closing stdin (not portable) or detach
            // We'll detach to avoid blocking program exit; user input thread will end when program exits
            input_thread.detach();
        }
    }

    // At the end of run, flush the in-memory nn_index cache to disk if dirty
    try {
        if (nn_index_dirty && !nn_index_target_dir.empty()) {
            std::filesystem::path p(nn_index_target_dir);
            std::filesystem::create_directories(p);
            p /= "nn_index.json";
            std::ofstream jf(p.string(), std::ofstream::out | std::ofstream::trunc);
            if (jf.is_open()) {
                jf << "{\n";
                size_t cnt = 0;
                for (const auto &kv: nn_index_cache) {
                    jf << "  \"" << kv.first << "\": " << kv.second;
                    ++cnt;
                    if (cnt < nn_index_cache.size()) jf << ",\n"; else jf << "\n";
                }
                jf << "}\n";
                jf.close();
            }
            nn_index_dirty = false;
        }
    } catch (...) {
        // ignore flush errors
    }
}

void Simulation::visualize(size_t cols) const {
    std::cout << get_visualization(cols) << "\n";
}

std::string Simulation::get_visualization(size_t cols, size_t height) const {
    // Build two stacked vertical bar plots: resources (height rows) then population (pop_height rows)
    const size_t pop_height = 4;
    std::vector<double> bins_res(cols, 0.0);
    std::vector<int> bins_pop(cols, 0);

    // compute per-bin resource averages and population counts
    for (size_t c=0;c<cols;++c) {
        size_t start = (c * env.width) / cols;
        size_t end = ((c+1) * env.width) / cols;
        double sumres = 0.0;
        int sumpop = 0;
        for (size_t i=start;i<end && i<env.width;++i) {
            sumres += env.resources[i];
        }
        bins_res[c] = sumres / std::max<size_t>(1, end-start);
    }

    // count individuals per bin
    for (auto &ind: individuals) {
        size_t bin = (ind.position * cols) / env.width;
        if (bin >= cols) bin = cols-1;
        bins_pop[bin]++;
    }

    double mx_res = 0.0;
    for (double v: bins_res) if (v > mx_res) mx_res = v;
    if (mx_res <= 0.0) mx_res = 1.0;

    int mx_pop = 1;
    for (int v: bins_pop) if (v > mx_pop) mx_pop = v;

    std::string out;

    // Resource plot (height rows)
    for (size_t row=0; row<height; ++row) {
        double threshold = mx_res * (double)(height - row) / (double)height;
        for (size_t c=0;c<cols;++c) {
            out.push_back(bins_res[c] >= threshold ? '#' : ' ');
        }
        out.push_back('\n');
    }

    // separator line
    for (size_t c=0;c<cols;++c) out.push_back('-');
    out.push_back('\n');

    // Population plot (pop_height rows)
    for (size_t row=0; row<pop_height; ++row) {
        int threshold = (int)std::ceil(mx_pop * (double)(pop_height - row) / (double)pop_height);
        for (size_t c=0;c<cols;++c) {
            out.push_back(bins_pop[c] >= threshold ? 'X' : ' ');
        }
        out.push_back('\n');
    }

    // bottom numeric population line (0-9)
    for (size_t c=0;c<cols;++c) {
        int p = bins_pop[c];
        if (p > 9) out.push_back('9'); else out.push_back(char('0' + std::min(9, p)));
    }
    out.push_back('\n');

    return out;
}
