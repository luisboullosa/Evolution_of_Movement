use crate::config::Config;
use crate::environment::Environment;
use crate::individual::Individual;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;
use rand_distr::Distribution;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

pub struct Simulation {
    pub env: Environment,
    pub individuals: Vec<Individual>,
    pub rng: StdRng,
    pub next_id: i32,
    pub last_spend: Vec<f64>,
    pub last_delta: Vec<i32>,
    pub last_movers: usize,
    pub config: Config,
    pub nn_index_cache: HashMap<String, String>,
    pub nn_index_dirty: bool,
    pub nn_index_target_dir: String,
    // per-sim buffered writers for env and individuals to avoid reopen on each timestep
    pub env_file: Option<File>,
    pub ind_file: Option<File>,
    // reusable NN buffers to avoid per-step allocations
    pub nn_hidden_buf: Vec<f64>,
    pub nn_out2_buf: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_init_population() {
        let mut cfg = Config::default();
        cfg.grid_width = 10;
        cfg.nn_nodes = 2;
        cfg.nn_inputs = 1;
        cfg.nn_layers = 1;
        cfg.nn_nodes2 = 0;
        cfg.randomize_start_positions = true;
        cfg.radiate = 0;
        cfg.seed = 42;

        let mut sim = Simulation::new(&cfg);
        assert_eq!(sim.individuals.len(), 0);
        sim.init_population(5);
        assert_eq!(sim.individuals.len(), 5);
        assert_eq!(sim.next_id, 6);
        for ind in sim.individuals.iter() {
            assert!(ind.position < cfg.grid_width as usize);
            assert!(ind.resources > 0.0);
        }
    }

    #[test]
    fn step_and_visualization_runs() {
        let mut cfg = Config::default();
        cfg.grid_width = 8;
        cfg.nn_nodes = 2;
        cfg.nn_inputs = 1;
        cfg.nn_layers = 1;
        cfg.nn_nodes2 = 0;
        cfg.randomize_start_positions = true;
        cfg.radiate = 0;
        cfg.seed = 7;
        cfg.vis_height = 6;

        let mut sim = Simulation::new(&cfg);
        sim.init_population(3);
        sim.step();
        let out = sim.get_visualization(4, cfg.vis_height as usize);
        assert!(!out.is_empty());
    }
}

impl Simulation {
    pub fn new(cfg: &Config) -> Self {
        let mut env = Environment::new(cfg.grid_width);
        env.init_perlin(cfg.seed as u64, cfg.resource_scale);
        Simulation {
            env,
            individuals: Vec::new(),
            rng: StdRng::seed_from_u64(cfg.seed as u64),
            next_id: 1,
            last_spend: Vec::new(),
            last_delta: Vec::new(),
            last_movers: 0,
            config: cfg.clone(),
            nn_index_cache: HashMap::new(),
            nn_index_dirty: false,
            nn_index_target_dir: String::new(),
            env_file: None,
            ind_file: None,
            nn_hidden_buf: vec![0.0; cfg.nn_nodes as usize],
            nn_out2_buf: vec![0.0; cfg.nn_nodes2 as usize],
        }
    }

    pub fn init_population(&mut self, n: usize) {
        use rand::Rng;
        let pdist = rand::distributions::Uniform::new(
            0usize,
            if self.env.width == 0 {
                1
            } else {
                self.env.width
            },
        );
        let wrand = rand::distributions::Uniform::new(-1.0f64, 1.0f64);
        for _ in 0..n {
            let mut nn = crate::neural_network::NeuralNetwork::new(
                self.config.nn_nodes as usize,
                self.config.nn_inputs as usize,
                self.config.nn_layers,
                self.config.nn_nodes2 as usize,
            );
            // initialize weights.
            // If `radiate` is set (>0) we start from default weights (zeros)
            // and apply simulated mutations at effective rate = mutation_rate * radiate.
            if self.config.radiate > 0 {
                let mut eff = (self.config.mutation_rate as f64) * (self.config.radiate as f64);
                if eff.is_nan() || eff < 0.0 {
                    eff = 0.0;
                }
                let normal = Normal::new(0.0, self.config.mutation_sigma).unwrap();
                for w in nn.weights.iter_mut() {
                    if self.rng.gen::<f64>() < eff {
                        *w += normal.sample(&mut self.rng);
                    }
                }
                for w in nn.weights_multi.iter_mut() {
                    if self.rng.gen::<f64>() < eff {
                        *w += normal.sample(&mut self.rng);
                    }
                }
                for b in nn.biases.iter_mut() {
                    if self.rng.gen::<f64>() < eff {
                        *b += normal.sample(&mut self.rng);
                    }
                }
                for w in nn.weights2.iter_mut() {
                    if self.rng.gen::<f64>() < eff {
                        *w += normal.sample(&mut self.rng);
                    }
                }
                for b in nn.biases2.iter_mut() {
                    if self.rng.gen::<f64>() < eff {
                        *b += normal.sample(&mut self.rng);
                    }
                }
            } else {
                // initialize weights randomly (uniform -1..1)
                for w in nn.weights.iter_mut() {
                    *w = self.rng.sample(wrand);
                }
                for w in nn.weights_multi.iter_mut() {
                    *w = self.rng.sample(wrand);
                }
                for b in nn.biases.iter_mut() {
                    *b = self.rng.sample(wrand);
                }
                for w in nn.weights2.iter_mut() {
                    *w = self.rng.sample(wrand);
                }
                for b in nn.biases2.iter_mut() {
                    *b = self.rng.sample(wrand);
                }
            }
            let pos = if self.config.randomize_start_positions {
                self.rng.sample(pdist)
            } else {
                self.config.start_pos
                    % (if self.env.width == 0 {
                        1
                    } else {
                        self.env.width
                    })
            };
            let ind = Individual::new(nn, pos, self.next_id);
            self.next_id += 1;
            self.individuals.push(ind);
        }
    }

    pub fn step(&mut self) {
        // advance environment dynamics
        self.env.step(
            self.config.seed as u64,
            self.config.env_time_speed,
            self.config.env_time_amplitude,
            self.config.env_perlin_spatial_scale,
            self.config.env_perlin_time_scale,
            self.config.resource_scale
        );

        // map positions to counts
        let clen = self.env.width.max(1);
        let mut counts = vec![0usize; clen];
        for ind in &self.individuals {
            counts[ind.position % clen] += 1;
        }

        // prepare per-individual tracking
        let n = self.individuals.len();
        self.last_spend.clear();
        self.last_spend.resize(n, 0.0);
        self.last_delta.clear();
        self.last_delta.resize(n, 0);
        self.last_movers = 0;

        // each individual acts and may move
        for i in 0..n {
            let ind = &mut self.individuals[i];
            let pos = ind.position % self.env.width.max(1);
            let center = self.env.gather_at(pos, counts[pos] as i32);
            let left = self.env.gather_at(
                (pos + self.env.width - 1) % self.env.width,
                counts[pos] as i32,
            );
            let right = self
                .env
                .gather_at((pos + 1) % self.env.width, counts[pos] as i32);
            let occupancy = counts[pos] as i32;
            let out = ind.act(
                left,
                center,
                right,
                occupancy,
                self.config.movement_cost_factor,
                &mut self.rng,
                &mut self.nn_hidden_buf,
                &mut self.nn_out2_buf,
            );
            if out != 0.0 {
                let delta = out.round() as i32;
                self.last_spend[i] = out.abs();
                self.last_delta[i] = delta;
                self.last_movers = self.last_movers.saturating_add(1);
                let mut new_position = ind.position as i32 + delta;
                while new_position < 0 {
                    new_position += self.env.width as i32;
                }
                new_position = (new_position as usize % self.env.width) as i32;
                ind.position = new_position as usize;
            }
        }

        // age increment
        for ind in &mut self.individuals {
            ind.age += 1;
        }

        // reproduction
        let before = self.individuals.len();
        for i in 0..before {
            if i >= self.individuals.len() {
                break;
            }
            // read resource value first to avoid borrowing conflicts
            let res = self.individuals[i].resources;
            if res > self.config.reproduction_threshold {
                if self.config.max_population > 0
                    && (self.individuals.len() as i32) >= self.config.max_population
                {
                    break;
                }
                // perform reproduction with a mutable borrow only when needed
                let child_opt = {
                    let ind_mut = &mut self.individuals[i];
                    ind_mut.reproduce(
                        &mut self.rng,
                        &mut self.next_id,
                        self.config.reproduction_threshold,
                        self.config.reproduction_cost,
                        self.config.mutation_rate,
                        self.config.mutation_sigma,
                    )
                };
                if let Some(child) = child_opt {
                    self.individuals.push(child);
                }
            }
        }

        // maintenance cost per individual
        for ind in &mut self.individuals {
            let total_nodes = ind.nn.nodes + ind.nn.nodes2;
            ind.resources -= (total_nodes as f64) * self.config.maintenance_cost_per_node;
        }

        // remove dead individuals
        use rand::distributions::{Distribution, Uniform};
        let ud = Uniform::new(0.0f64, 1.0f64);
        self.individuals.retain(|ind| {
            if ind.resources <= 0.0 {
                return false;
            }
            if self.config.death_risk > 0.0 && ind.age > 0 {
                let p = 1.0 - (-(self.config.death_risk * (ind.age as f64))).exp();
                if ud.sample(&mut self.rng) < p {
                    return false;
                }
            }
            true
        });
    }

    /// Append environment NDJSON line to `env.jsonl` in target dir derived from `target_path` or `config.webviewer_base`.
    pub fn dump_env(&self, _target_path: &str, t: Option<i32>) {
        // Skip any env output when global write_outputs is disabled
        if !self.config.write_outputs {
            return;
        }
        // If a target path was provided, use it directly (it may include filename).
        // Otherwise create/find a per-sim directory under `webviewer_base` and write there.
        let path = if !_target_path.is_empty() {
            std::path::PathBuf::from(_target_path)
        } else {
            let base = if !self.nn_index_target_dir.is_empty() {
                std::path::PathBuf::from(&self.nn_index_target_dir)
            } else {
                std::path::PathBuf::from(if !self.config.webviewer_base.is_empty() {
                    &self.config.webviewer_base
                } else {
                    "."
                })
            };
            let _ = std::fs::create_dir_all(&base);
            let mut simnum = self.config.sim_number;
            if simnum < 0 {
                let mut maxn = -1;
                if let Ok(entries) = std::fs::read_dir(&base) {
                    for e in entries.flatten() {
                        if let Ok(fname) = e.file_name().into_string() {
                            if fname.starts_with(&self.config.sim_prefix) {
                                let tail = &fname[self.config.sim_prefix.len()..];
                                if let Ok(n) = tail.parse::<i32>() {
                                    if n > maxn {
                                        maxn = n;
                                    }
                                }
                            }
                        }
                    }
                }
                simnum = maxn + 1;
            }
            let dn = format!(
                "{}{:0width$}",
                self.config.sim_prefix,
                simnum,
                width = self.config.sim_digits as usize
            );
            let simpath = base.join(&dn);
            let _ = std::fs::create_dir_all(&simpath);
            simpath.join("env.jsonl")
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = write!(f, "{{");
            if let Some(tv) = t {
                let _ = write!(f, "\"t\":{}, ", tv);
            }
            let _ = write!(f, "\"resources\": [");
            for i in 0..self.env.width {
                if i > 0 {
                    let _ = write!(f, ", ");
                }
                let _ = write!(f, "{:.6}", self.env.resources[i]);
            }
            let _ = writeln!(f, "]}}");
        }
    }

    /// Append individuals NDJSON line and update in-memory nn index cache.
    pub fn dump_individuals(&mut self, _target_path: &str, t: Option<i32>) {
        // Skip any individuals / nn index work when global write_outputs is disabled
        if !self.config.write_outputs {
            return;
        }
        // Respect provided target path (may include filename). Fall back to a per-sim folder under webviewer_base.
        let path = if !_target_path.is_empty() {
            std::path::PathBuf::from(_target_path)
        } else {
            let base = if !self.nn_index_target_dir.is_empty() {
                std::path::PathBuf::from(&self.nn_index_target_dir)
            } else {
                std::path::PathBuf::from(if !self.config.webviewer_base.is_empty() {
                    &self.config.webviewer_base
                } else {
                    "."
                })
            };
            let _ = std::fs::create_dir_all(&base);
            let mut simnum = self.config.sim_number;
            if simnum < 0 {
                let mut maxn = -1;
                if let Ok(entries) = std::fs::read_dir(&base) {
                    for e in entries.flatten() {
                        if let Ok(fname) = e.file_name().into_string() {
                            if fname.starts_with(&self.config.sim_prefix) {
                                let tail = &fname[self.config.sim_prefix.len()..];
                                if let Ok(n) = tail.parse::<i32>() {
                                    if n > maxn {
                                        maxn = n;
                                    }
                                }
                            }
                        }
                    }
                }
                simnum = maxn + 1;
            }
            let dn = format!(
                "{}{:0width$}",
                self.config.sim_prefix,
                simnum,
                width = self.config.sim_digits as usize
            );
            let simpath = base.join(&dn);
            let _ = std::fs::create_dir_all(&simpath);
            simpath.join("individuals.jsonl")
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = write!(f, "{{");
            if let Some(tv) = t {
                let _ = write!(f, "\"t\":{}, ", tv);
            }
            let _ = write!(f, "\"individuals\": [");
            for (i, ind) in self.individuals.iter().enumerate() {
                if i > 0 {
                    let _ = write!(f, ", ");
                }
                let spend = if i < self.last_spend.len() {
                    self.last_spend[i]
                } else {
                    0.0
                };
                let delta = if i < self.last_delta.len() {
                    self.last_delta[i]
                } else {
                    0
                };
                let _ = write!(f, "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"last_spend\":{:.6},\"last_delta\":{},\"age\":{} }}", ind.id, ind.position, ind.resources, spend, delta, ind.age);

                // nn index cache (build nested arrays for multi-input nets to match C++ format)
                let idstr = ind.id.to_string();
                use std::collections::hash_map::Entry;
                match self.nn_index_cache.entry(idstr) {
                    Entry::Occupied(_) => {}
                    Entry::Vacant(e) => {
                        let mut js = String::new();
                        js.push('{');
                        // type
                        let ntype = match ind.nn.nn_type {
                            crate::neural_network::NNType::SingleInput => "single",
                            _ => "multi",
                        };
                        js.push_str(&format!("\n      \"type\": \"{}\",\n", ntype));
                        js.push_str(&format!("      \"inputs\": {},\n      \"nodes\": {},\n      \"layers\": {},\n      \"nodes2\": {}\n", ind.nn.inputs, ind.nn.nodes, ind.nn.layers, ind.nn.nodes2));

                        // weights
                        js.push_str("      ,\"weights\": ");
                        if let crate::neural_network::NNType::SingleInput = ind.nn.nn_type {
                            js.push('[');
                            for (wi, w) in ind.nn.weights.iter().enumerate() {
                                if wi > 0 {
                                    js.push_str(", ");
                                }
                                js.push_str(&format!("{:.6}", w));
                            }
                            js.push_str("]\n");
                        } else {
                            js.push('[');
                            for ni in 0..ind.nn.nodes {
                                if ni > 0 {
                                    js.push_str(", ");
                                }
                                js.push('[');
                                for j in 0..ind.nn.inputs.min(ind.nn.weights_multi.len() / ind.nn.nodes)
                                {
                                    if j > 0 {
                                        js.push_str(", ");
                                    }
                                    let idx = ni.saturating_mul(ind.nn.inputs).saturating_add(j);
                                    js.push_str(&format!(
                                        "{:.6}",
                                        ind.nn.weights_multi.get(idx).copied().unwrap_or(0.0)
                                    ));
                                }
                                js.push(']');
                            }
                            js.push_str("]\n");
                        }

                        // biases
                        js.push_str(",      \"biases\": [");
                        for (bi, b) in ind.nn.biases.iter().enumerate() {
                            if bi > 0 {
                                js.push_str(", ");
                            }
                            js.push_str(&format!("{:.6}", b));
                        }
                        js.push_str("]\n");

                        // optional second layer
                        if ind.nn.layers == 2 {
                            js.push_str(",      \"layer2_weights\": [");
                            for ni in 0..ind.nn.nodes2 {
                                if ni > 0 {
                                    js.push_str(", ");
                                }
                                js.push('[');
                                for j in 0..ind.nn.nodes {
                                    if j > 0 {
                                        js.push_str(", ");
                                    }
                                    let idx = ni.saturating_mul(ind.nn.nodes).saturating_add(j);
                                    js.push_str(&format!(
                                        "{:.6}",
                                        ind.nn.weights2.get(idx).copied().unwrap_or(0.0)
                                    ));
                                }
                                js.push(']');
                            }
                            js.push_str("]\n");
                            js.push_str(",      \"layer2_biases\": [");
                            for (bi, b) in ind.nn.biases2.iter().enumerate() {
                                if bi > 0 {
                                    js.push_str(", ");
                                }
                                js.push_str(&format!("{:.6}", b));
                            }
                            js.push_str("]\n");
                        }

                        js.push_str("    }");
                        e.insert(js);
                        self.nn_index_dirty = true;
                    }
                }
            }
            let _ = writeln!(f, "]}}");
        }
    }

    /// Append environment NDJSON line into `simdir` using filename from config (or default).
    pub fn dump_env_to_dir(&mut self, simdir: &str, t: Option<i32>) {
        // Skip writing env to per-sim dir when outputs disabled
        if !self.config.write_outputs {
            return;
        }
        let simpath = std::path::Path::new(simdir);
        let env_fname = std::path::Path::new(&self.config.env_out_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("env.jsonl");
        let path = simpath.join(env_fname);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Open once per simulation and reuse the file handle
        if self.env_file.is_none() {
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => {
                    self.env_file = Some(f);
                }
                Err(_e) => {
                    // ignore failures to open env file
                }
            }
        }
        if let Some(f) = self.env_file.as_mut() {
            let _ = write!(f, "{{");
            if let Some(tv) = t {
                let _ = write!(f, "\"t\":{}, ", tv);
            }
            let _ = write!(f, "\"resources\": [");
            for i in 0..self.env.width {
                if i > 0 {
                    let _ = write!(f, ", ");
                }
                let _ = write!(f, "{:.6}", self.env.resources[i]);
            }
            let _ = writeln!(f, "]}}");
            let _ = f.flush();
        }
    }

    /// Append individuals NDJSON line into `simdir` and update nn index cache.
    pub fn dump_individuals_to_dir(&mut self, simdir: &str, t: Option<i32>) {
        // Skip writing individuals/nn index to per-sim dir when outputs disabled
        if !self.config.write_outputs {
            return;
        }
        let simpath = std::path::Path::new(simdir);
        let ind_fname = std::path::Path::new(&self.config.individuals_out_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("individuals.jsonl");
        let path = simpath.join(ind_fname);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Open once per simulation and reuse the file handle
        if self.ind_file.is_none() {
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => {
                    self.ind_file = Some(f);
                }
                Err(_e) => {
                    // ignore failures to open individuals file
                }
            }
        }
        if let Some(f) = self.ind_file.as_mut() {
            let _ = write!(f, "{{");
            if let Some(tv) = t {
                let _ = write!(f, "\"t\":{}, ", tv);
            }
            let _ = write!(f, "\"individuals\": [");
            for (i, ind) in self.individuals.iter().enumerate() {
                if i > 0 {
                    let _ = write!(f, ", ");
                }
                let spend = if i < self.last_spend.len() {
                    self.last_spend[i]
                } else {
                    0.0
                };
                let delta = if i < self.last_delta.len() {
                    self.last_delta[i]
                } else {
                    0
                };
                let _ = write!(f, "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"last_spend\":{:.6},\"last_delta\":{},\"age\":{} }}", ind.id, ind.position, ind.resources, spend, delta, ind.age);

                // nn index cache (build nested arrays for multi-input nets to match C++ format)
                let idstr = ind.id.to_string();
                use std::collections::hash_map::Entry;
                match self.nn_index_cache.entry(idstr) {
                    Entry::Occupied(_) => {}
                    Entry::Vacant(e) => {
                        let mut js = String::new();
                        js.push('{');
                        // type
                        let ntype = match ind.nn.nn_type {
                            crate::neural_network::NNType::SingleInput => "single",
                            _ => "multi",
                        };
                        js.push_str(&format!("\n      \"type\": \"{}\",\n", ntype));
                        js.push_str(&format!("      \"inputs\": {},\n      \"nodes\": {},\n      \"layers\": {},\n      \"nodes2\": {}\n", ind.nn.inputs, ind.nn.nodes, ind.nn.layers, ind.nn.nodes2));

                        // weights
                        js.push_str("      ,\"weights\": ");
                        if let crate::neural_network::NNType::SingleInput = ind.nn.nn_type {
                            js.push('[');
                            for (wi, w) in ind.nn.weights.iter().enumerate() {
                                if wi > 0 {
                                    js.push_str(", ");
                                }
                                js.push_str(&format!("{:.6}", w));
                            }
                            js.push_str("]\n");
                        } else {
                            js.push('[');
                            for ni in 0..ind.nn.nodes {
                                if ni > 0 {
                                    js.push_str(", ");
                                }
                                js.push('[');
                                for j in 0..ind.nn.inputs.min(ind.nn.weights_multi.len() / ind.nn.nodes) {
                                    if j > 0 {
                                        js.push_str(", ");
                                    }
                                    let idx = ni.saturating_mul(ind.nn.inputs).saturating_add(j);
                                    js.push_str(&format!("{:.6}", ind.nn.weights_multi.get(idx).copied().unwrap_or(0.0)));
                                }
                                js.push(']');
                            }
                            js.push_str("]\n");
                        }
                        js.push_str(",      \"biases\": [");
                        for (bi, b) in ind.nn.biases.iter().enumerate() {
                            if bi > 0 {
                                js.push_str(", ");
                            }
                            js.push_str(&format!("{:.6}", b));
                        }
                        js.push_str("]\n");

                        // optional second layer
                        if ind.nn.layers == 2 {
                            js.push_str(",      \"layer2_weights\": [");
                            for ni in 0..ind.nn.nodes2 {
                                if ni > 0 {
                                    js.push_str(", ");
                                }
                                js.push('[');
                                for j in 0..ind.nn.nodes {
                                    if j > 0 {
                                        js.push_str(", ");
                                    }
                                    let idx = ni.saturating_mul(ind.nn.nodes).saturating_add(j);
                                    js.push_str(&format!("{:.6}", ind.nn.weights2.get(idx).copied().unwrap_or(0.0)));
                                }
                                js.push(']');
                            }
                            js.push_str("]\n");
                            js.push_str(",      \"layer2_biases\": [");
                            for (bi, b) in ind.nn.biases2.iter().enumerate() {
                                if bi > 0 {
                                    js.push_str(", ");
                                }
                                js.push_str(&format!("{:.6}", b));
                            }
                            js.push_str("]\n");
                        }

                        js.push_str("    }");
                        e.insert(js);
                        self.nn_index_dirty = true;
                    }
                }
            }
            let _ = writeln!(f, "]}}");
            let _ = f.flush();
        }
    }

    pub fn get_visualization(&self, cols: usize, height: usize) -> String {
        // simplified copy of C++ visualization
        let pop_height = 4usize;
        let mut bins_res = vec![0.0f64; cols];
        let mut bins_pop = vec![0usize; cols];
        for (c, slot) in bins_res.iter_mut().enumerate().take(cols) {
            let start = (c * self.env.width) / cols;
            let end = ((c + 1) * self.env.width) / cols;
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for i in start..end.min(self.env.width) {
                sum += self.env.resources[i];
                cnt += 1;
            }
            *slot = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
        }
        for ind in &self.individuals {
            let mut bin = (ind.position * cols) / self.env.width.max(1);
            if bin >= cols {
                bin = cols - 1;
            }
            bins_pop[bin] += 1;
        }
        let mx_res = bins_res.iter().cloned().fold(f64::NAN, f64::max).max(1.0);
        let mx_pop = *bins_pop.iter().max().unwrap_or(&1);
        let mut out = String::new();
        for row in 0..height {
            let threshold = mx_res * (height - row) as f64 / height as f64;
            for &res in bins_res.iter().take(cols) {
                out.push(if res >= threshold { '#' } else { ' ' });
            }
            out.push('\n');
        }
        out.push_str(&"-".repeat(cols));
        out.push('\n');
        for row in 0..pop_height {
            let threshold =
                ((mx_pop as f64) * (pop_height - row) as f64 / pop_height as f64).ceil() as usize;
            for &count in bins_pop.iter().take(cols) {
                out.push(if count >= threshold { 'X' } else { ' ' });
            }
            out.push('\n');
        }
        for &p in bins_pop.iter().take(cols) {
            out.push(char::from_digit(p.min(9) as u32, 10).unwrap_or('0'));
        }
        out.push('\n');
        out
    }

    pub fn visualize(&self, cols: usize) {
        println!("{}", self.get_visualization(cols, self.config.vis_height));
    }

    pub fn run(&mut self) {
        use std::fs;
        use std::io::{self, Write};
        let steps = self.config.steps;
        let report_interval = if self.config.report_interval > 0 {
            self.config.report_interval
        } else {
            1
        };
        let write_frames = self.config.write_frames;
        // no interactive stdin stop handler — progress is non-interactive

        // Compute and create per-sim directory up front only when outputs are requested.
        let mut fout: Option<std::fs::File> = None;
        let simdir: String;
        if self.config.write_outputs {
            // Resolve `webviewer_base` against the repository root when it's a relative path.
            // Repository root is assumed to contain `CMakeLists.txt` or `README.md`.
            fn find_repo_root(mut cur: std::path::PathBuf) -> Option<std::path::PathBuf> {
                loop {
                    if cur.join("CMakeLists.txt").exists() || cur.join("README.md").exists() {
                        return Some(cur);
                    }
                    if let Some(p) = cur.parent() {
                        cur = p.to_path_buf();
                    } else {
                        return None;
                    }
                }
            }

            let base = if !self.config.webviewer_base.is_empty() {
                let wb = std::path::Path::new(&self.config.webviewer_base);
                if wb.is_absolute() {
                    wb.to_path_buf()
                } else {
                    // prefer repo root if available, otherwise cwd
                    if let Ok(cwd) = std::env::current_dir() {
                        if let Some(root) = find_repo_root(cwd.clone()) {
                            root.join(wb)
                        } else {
                            cwd.join(wb)
                        }
                    } else {
                        wb.to_path_buf()
                    }
                }
            } else {
                // default to repo root if available, else current dir
                if let Ok(cwd) = std::env::current_dir() {
                    if let Some(root) = find_repo_root(cwd.clone()) {
                        root
                    } else {
                        cwd
                    }
                } else {
                    std::path::PathBuf::from(".")
                }
            };
            let _ = fs::create_dir_all(&base);
            // determine sim number (respect negative -> auto-increment)
            let mut simnum = self.config.sim_number;
            if simnum < 0 {
                let mut maxn = -1;
                if let Ok(entries) = fs::read_dir(&base) {
                    for e in entries.flatten() {
                        if let Ok(fname) = e.file_name().into_string() {
                            if fname.starts_with(&self.config.sim_prefix) {
                                let tail = &fname[self.config.sim_prefix.len()..];
                                if let Ok(n) = tail.parse::<i32>() {
                                    if n > maxn {
                                        maxn = n;
                                    }
                                }
                            }
                        }
                    }
                }
                simnum = maxn + 1;
            }
            let dn = format!(
                "{}{:0width$}",
                self.config.sim_prefix,
                simnum,
                width = self.config.sim_digits as usize
            );
            let simpath = base.join(&dn);
            let _ = fs::create_dir_all(&simpath);
            simdir = simpath.to_string_lossy().to_string();

            // Ensure nn_index_target_dir prefers the sim directory so index flushes
            if self.nn_index_target_dir.is_empty() {
                self.nn_index_target_dir = simdir.clone();
            }

            // If frames are enabled, create the frames file inside simdir
            // Respect `write_outputs` to allow skipping all per-step outputs.
            if write_frames && self.config.write_outputs {
                let fpath = simpath.join(&self.config.frames_filename);
                if let Ok(f) = fs::File::create(&fpath) {
                    fout = Some(f);
                }
                // write header and optional config
                if let Some(f) = fout.as_mut() {
                    if self.config.write_config {
                        let js =
                            serde_json::to_string(&self.config).unwrap_or_else(|_| "{}".to_string());
                        let _ = writeln!(f, "#CONFIG {}", js);

                        // also write a machine-readable config.json inside the sim directory
                        // using an atomic write (tmp file + rename) so readers never see
                        // a partial file.
                        let mut cfg_tmp = simpath.clone();
                        cfg_tmp.push("config.json.tmp");
                        if let Ok(mut cout) = fs::File::create(&cfg_tmp) {
                            let _ = write!(cout, "{}", js);
                            let _ = cout.flush();
                            let mut finalp = std::path::PathBuf::from(&simpath);
                            finalp.push("config.json");
                            let _ = std::fs::rename(&cfg_tmp, &finalp);
                        }

                        // if a custom config_out_path was requested, write that atomically too
                        if !self.config.config_out_path.is_empty() {
                            let custom = simpath.join(&self.config.config_out_path);
                            if let Some(parent) = custom.parent() {
                                let _ = fs::create_dir_all(parent);
                            }
                            let mut custom_tmp = custom.clone();
                            if let Some(fname) = custom.file_name() {
                                let tmpname = format!("{}.tmp", fname.to_string_lossy());
                                custom_tmp.set_file_name(tmpname);
                            } else {
                                custom_tmp.push(".tmp");
                            }
                            if let Ok(mut coutf) = fs::File::create(&custom_tmp) {
                                let _ = write!(coutf, "{}", js);
                                let _ = coutf.flush();
                                let _ = std::fs::rename(&custom_tmp, &custom);
                            }
                        }
                    }
                    let _ = writeln!(f, "t\tpop\tavg_res\tframe\tprop_moved");

                    // write initial env/individuals snapshot so downstream tools/tests
                    // can find a starting state even if burn_in prevents periodic dumps.
                    if self.config.write_outputs {
                        if !simdir.is_empty() {
                            self.dump_env_to_dir(&simdir, Some(0));
                            self.dump_individuals_to_dir(&simdir, Some(0));
                        } else {
                            let env_out = self.config.env_out_path.clone();
                            let ind_out = self.config.individuals_out_path.clone();
                            self.dump_env(&env_out, Some(0));
                            self.dump_individuals(&ind_out, Some(0));
                        }
                    }
                }
            }
        } else {
            // outputs disabled -> no per-sim directory
            simdir = String::new();
        }

        let start_time = Instant::now();
        for t in 0..steps {
            self.step();

            if self.individuals.is_empty() {
                println!("[info] population extinct at t={}, stopping simulation", t);
                break;
            }

            if self.config.show_progress {
                // progress bar + ETA
                let bar_width = 40usize;
                let frac = (t + 1) as f64 / steps as f64;
                let pos = (bar_width as f64 * frac) as usize;
                let mut ss = String::new();
                ss.push('[');
                for i in 0..bar_width {
                    ss.push(if i < pos {
                        '='
                    } else if i == pos {
                        '>'
                    } else {
                        ' '
                    });
                }
                ss.push(']');
                ss.push(' ');
                ss.push_str(&format!("{}%", (frac * 100.0) as i32));
                // ETA
                let elapsed = start_time.elapsed().as_secs_f64();
                let avg = elapsed / ((t + 1) as f64);
                let rem = ((steps - (t + 1)) as f64) * avg;
                let hrs = (rem as i64) / 3600;
                let mins = ((rem as i64) % 3600) / 60;
                let secs = (rem as i64) % 60;
                ss.push_str(&format!("  ETA {}:{:02}:{:02}", hrs, mins, secs));

                // prop moved
                let mut prop_now = 0.0f64;
                if !self.last_delta.is_empty() {
                    let seen = std::cmp::min(self.last_delta.len(), self.individuals.len());
                    if seen > 0 {
                        let moved = self
                            .last_delta
                            .iter()
                            .take(seen)
                            .filter(|&&d| d != 0)
                            .count();
                        prop_now = moved as f64 / seen as f64;
                    }
                }
                ss.push_str(&format!(
                    "  t={} pop={} prop_moved={:.3}",
                    t,
                    self.individuals.len(),
                    prop_now
                ));
                ss.push('\r');
                print!("{}", ss);
                let _ = io::stdout().flush();
            }

            if t % report_interval == 0 {
                if t < self.config.burn_in {
                    continue;
                }

                let mut avgres = 0.0f64;
                for ind in &self.individuals {
                    avgres += ind.resources;
                }
                if !self.individuals.is_empty() {
                    avgres /= self.individuals.len() as f64;
                }
                let prop_moved = if !self.last_delta.is_empty() {
                    let seen = std::cmp::min(self.last_delta.len(), self.individuals.len());
                    if seen > 0 {
                        self.last_delta
                            .iter()
                            .take(seen)
                            .filter(|&&d| d != 0)
                            .count() as f64
                            / seen as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // visualization
                let frame = self.get_visualization(self.config.vis_cols, self.config.vis_height);
                // Do not print frames to console by default (writes to frames.tsv instead).
                if let Some(f) = fout.as_mut() {
                    let mut esc = String::new();
                    for ch in frame.chars() {
                        if ch == '\n' {
                            esc.push_str("\\n");
                        } else {
                            esc.push(ch);
                        }
                    }
                    let _ = writeln!(
                        f,
                        "{}\t{}\t{}\t{}\t{}",
                        t,
                        self.individuals.len(),
                        avgres,
                        esc,
                        prop_moved
                    );
                }

                // dump env/individuals into a single JSONL file per sim (no per-timepoint suffix)
                if self.config.write_outputs {
                    if !simdir.is_empty() {
                        self.dump_env_to_dir(&simdir, Some(t));
                        self.dump_individuals_to_dir(&simdir, Some(t));
                    } else {
                        let env_out = self.config.env_out_path.clone();
                        let ind_out = self.config.individuals_out_path.clone();
                        self.dump_env(&env_out, Some(t));
                        self.dump_individuals(&ind_out, Some(t));
                    }
                }

                // periodic nn_index flush (skip when outputs disabled)
                if self.config.write_outputs && self.nn_index_dirty {
                    let do_flush = (self.config.nn_index_flush_interval > 0
                        && t % self.config.nn_index_flush_interval == 0)
                        || (self.config.nn_index_flush_size > 0
                            && self.nn_index_cache.len() >= self.config.nn_index_flush_size);
                        if do_flush {
                            let target_dir = if !self.nn_index_target_dir.is_empty() {
                                &self.nn_index_target_dir
                            } else if !simdir.is_empty() {
                                &simdir
                            } else {
                                &self.config.webviewer_base
                            };
                            let mut p = std::path::PathBuf::from(target_dir);
                            let _ = fs::create_dir_all(&p);
                            // write to temporary file then rename for atomicity
                            p.push("nn_index.json.tmp");
                            if let Ok(mut jf) = fs::File::create(&p) {
                                let _ = writeln!(jf, "{{");
                                let mut first = true;
                                for (k, v) in &self.nn_index_cache {
                                    if !first {
                                        let _ = writeln!(jf, ",");
                                    }
                                    first = false;
                                    let _ = write!(jf, "\"{}\": {}", k, v);
                                }
                                let _ = writeln!(jf, "}}");
                                let _ = jf.flush();
                                // rename tmp -> nn_index.json
                                let mut finalp = std::path::PathBuf::from(target_dir);
                                finalp.push("nn_index.json");
                                let _ = std::fs::rename(&p, &finalp);
                                self.nn_index_dirty = false;
                            }
                        }
                }
            }

            // step_mode debug pause
            if self.config.step_mode {
                self.print_debug_snapshot(3);
                if std::env::var("DISABLE_STEP_PAUSE").is_err() {
                    println!(
                        "Completed step {}/{} — press Enter to continue...",
                        t + 1,
                        steps
                    );
                    let _ = io::stdout().flush();
                    let mut buf = String::new();
                    let _ = io::stdin().read_line(&mut buf);
                }
            }
        }

        if let Some(mut f) = fout {
            let _ = f.flush();
        }
        if self.config.show_progress {
            println!();
        }

        // final nn_index flush on exit (only when outputs enabled)
        if self.config.write_outputs && self.nn_index_dirty && !self.nn_index_cache.is_empty() {
            let target_dir = if !self.nn_index_target_dir.is_empty() {
                &self.nn_index_target_dir
            } else if !simdir.is_empty() {
                &simdir
            } else {
                &self.config.webviewer_base
            };
            let mut p = std::path::PathBuf::from(target_dir);
            let _ = fs::create_dir_all(&p);
            // atomic write via temp file
            p.push("nn_index.json.tmp");
            if let Ok(mut jf) = fs::File::create(&p) {
                let _ = writeln!(jf, "{{");
                let mut first = true;
                for (k, v) in &self.nn_index_cache {
                    if !first {
                        let _ = writeln!(jf, ",");
                    }
                    first = false;
                    let _ = write!(jf, "\"{}\": {}", k, v);
                }
                let _ = writeln!(jf, "}}");
                let _ = jf.flush();
                let mut finalp = std::path::PathBuf::from(target_dir);
                finalp.push("nn_index.json");
                let _ = std::fs::rename(&p, &finalp);
                self.nn_index_dirty = false;
            }
        }

        // Ensure final env/individuals snapshot exists for consumers/tests
        if !simdir.is_empty() {
            self.dump_env_to_dir(&simdir, Some(steps));
            self.dump_individuals_to_dir(&simdir, Some(steps));
        } else {
            let env_out = self.config.env_out_path.clone();
            let ind_out = self.config.individuals_out_path.clone();
            self.dump_env(&env_out, Some(steps));
            self.dump_individuals(&ind_out, Some(steps));
        }

        // Emit a compact final summary to stdout so external runners (Python driver)
        // can capture final metrics even when per-step outputs are disabled.
        // This prints a single JSON object with `final_population` and `prop_moved`.
        let final_pop = self.individuals.len();
        let prop_moved = if !self.last_delta.is_empty() {
            let seen = std::cmp::min(self.last_delta.len(), self.individuals.len());
            if seen > 0 {
                self.last_delta
                    .iter()
                    .take(seen)
                    .filter(|&&d| d != 0)
                    .count() as f64
                    / seen as f64
            } else {
                0.0
            }
        } else {
            0.0
        };
        let mut summary_map = serde_json::Map::new();
        summary_map.insert("final_population".to_string(), serde_json::Value::from(final_pop));
        summary_map.insert("prop_moved".to_string(), serde_json::Value::from(prop_moved));
        if let Ok(s) = serde_json::to_string(&summary_map) {
            println!("{}", s);
        }
    }

    /// Produce a compact summary string for quick inspection in debugger or logs.
    pub fn debug_snapshot(&self, sample_count: usize) -> String {
        let mut s = format!(
            "Simulation snapshot: individuals={} env_width={}\n",
            self.individuals.len(),
            self.env.width
        );
        let n = sample_count.min(self.individuals.len());
        for i in 0..n {
            let ind = &self.individuals[i];
            s.push_str(&format!("  {}\n", ind.debug_summary()));
            s.push_str(&format!(
                "    weights_preview={:?}\n",
                ind.nn.weights_preview(5)
            ));
        }
        s
    }

    /// Print the debug snapshot to stdout.
    pub fn print_debug_snapshot(&self, sample_count: usize) {
        println!("{}", self.debug_snapshot(sample_count));
    }
}
