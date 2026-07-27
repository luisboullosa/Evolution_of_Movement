use crate::architecture_lineage::{ArchitectureLineageEvent, write_lineage_jsonl};
use crate::cladogram::Cladogram;
use crate::config::Config;
use crate::environment::Environment;
use crate::hex_dish::HexDishLayout;
use crate::hex_simulation::{grid_cell_count, step_individuals_cpu};
use crate::individual::Individual;
use crate::population_dynamics::{
    apply_metabolism_and_cull, clamp_initial_population, record_births_in_cladogram,
    record_births_in_lineage, run_reproduction_phase,
};
use crate::step_engine::apply_decisions_sequential;
use crate::run_record::{persist_run_record, RunRecordContext};
use crate::snapshot::{PopulationSnapshot, SnapshotError};
use crate::snapshot_registry;
use crate::tensor::{run_brain_forward, GpuForwardState, PopulationBrainTensor, RowRngManager};
use crate::webviewer_writer::{resolve_individuals_encoding, IndividualsEncoding, WebviewerWriter};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::time::Instant;

/// How often to redraw the progress bar (~50 updates max; respects `report_interval`).
fn progress_update_interval(steps: i32, report_interval: i32) -> i32 {
    let steps = steps.max(1);
    let min_interval = (steps / 50).max(1);
    let interval = if report_interval > 0 {
        report_interval.max(min_interval)
    } else {
        min_interval
    };
    interval.min(steps)
}

fn should_emit_progress(t: i32, steps: i32, interval: i32) -> bool {
    t == 0 || t + 1 >= steps || t % interval == 0
}

pub struct Simulation {
    pub env: Environment,
    pub individuals: Vec<Individual>,
    pub hex_dish: Option<Arc<HexDishLayout>>,
    pub rng: StdRng,
    pub next_id: i32,
    pub last_spend: Vec<f64>,
    pub last_delta: Vec<i32>,
    pub last_movers: usize,
    pub config: Config,
    pub nn_hidden_buf: Vec<f64>,
    pub nn_out2_buf: Vec<f64>,
    pub tensor: PopulationBrainTensor,
    pub row_rng: RowRngManager,
    pub gpu_state: GpuForwardState,
    pub step_count: i32,
    /// Brain forward passes executed on GPU during this run (for run_record).
    pub gpu_steps_executed: u64,
    /// Brain forward passes executed on CPU during this run (for run_record).
    pub cpu_brain_steps_executed: u64,
    ind_encoding: IndividualsEncoding,
    pub cladogram: Option<Cladogram>,
    architecture_lineage: Vec<ArchitectureLineageEvent>,
}

impl Simulation {
    pub fn new(cfg: &Config) -> Self {
        let config = cfg.clone();
        let hex_dish = if config.hex_dish_radius > 0 {
            Some(Arc::new(HexDishLayout::from_radius_options(
                config.hex_dish_radius,
                config.hex_dish_torus,
            )))
        } else {
            None
        };
        let cell_count = grid_cell_count(&config);
        let mut env = Environment::new(cell_count);
        if let Some(ref dish) = hex_dish {
            env.init_perlin_hex_dish(
                dish,
                config.seed as u64,
                config.resource_scale,
                config.env_perlin_spatial_scale,
                config.env_perlin_time_scale,
                config.resource_contrast,
            );
        } else {
            env.init_perlin(
                config.seed as u64,
                config.resource_scale,
                config.resource_contrast,
            );
        }
        let ind_encoding = Self::resolve_individuals_encoding(&config.individuals_encoding);
        let max_buf = config.max_brain_nodes.max(config.nn_nodes).max(config.nn_nodes2) as usize;
        let tensor = PopulationBrainTensor::new(&config);
        let row_rng = RowRngManager::new(config.seed as u64);
        Simulation {
            env,
            hex_dish,
            individuals: Vec::new(),
            rng: StdRng::seed_from_u64(config.seed as u64),
            next_id: 1,
            last_spend: Vec::new(),
            last_delta: Vec::new(),
            last_movers: 0,
            config,
            nn_hidden_buf: vec![0.0; max_buf.max(32)],
            nn_out2_buf: vec![0.0; max_buf.max(32)],
            tensor,
            row_rng,
            gpu_state: GpuForwardState::new(),
            step_count: 0,
            gpu_steps_executed: 0,
            cpu_brain_steps_executed: 0,
            ind_encoding,
            cladogram: if cfg.export_cladogram {
                Some(Cladogram::new())
            } else {
                None
            },
            architecture_lineage: Vec::new(),
        }
    }

    fn use_gpu_step(&self) -> bool {
        self.config.use_gpu_step && self.tensor.population_is_gpu_compatible()
    }

    pub fn init_population(&mut self, n: usize) {
        self.init_population_with_founder_mode(n, None);
    }

    pub fn init_population_with_founder_mode(
        &mut self,
        n: usize,
        founder_mode: Option<crate::brain::FounderWeightMode>,
    ) {
        use rand_distr::{Distribution, Uniform};
        let requested = n;
        let n = clamp_initial_population(n, self.config.max_population);
        if n < requested {
            eprintln!(
                "[warn] clamping initial population from {requested} to {n} (max_population={})",
                self.config.max_population
            );
        }
        let cell_count = grid_cell_count(&self.config);
        let pdist = Uniform::new(0usize, cell_count.max(1));
        for _ in 0..n {
            let pos = if self.config.randomize_start_positions {
                pdist.sample(&mut self.rng)
            } else {
                self.config.start_pos.min(cell_count.saturating_sub(1))
            };
            let mut ind = if let Some(mode) = founder_mode {
                Individual::new_founder_with_mode(&self.config, pos, self.next_id, &mut self.rng, mode)
            } else {
                Individual::new_founder(&self.config, pos, self.next_id, &mut self.rng)
            };
            if let Some(ref mut clado) = self.cladogram {
                clado.add_founder(ind.id, ind.lineage_color);
            }
            self.next_id += 1;
            ind.resources = 10.0;
            self.individuals.push(ind);
        }
        self.tensor.record_active(self.individuals.len());
    }

    pub fn step_with_existing_resources(&mut self) {
        self.step_count = self.step_count.saturating_add(1);
        if let Some(ref mut clado) = self.cladogram {
            clado.advance_time();
        }

        if self.use_gpu_step() {
            let cell_count = grid_cell_count(&self.config);
            let mut occupancy = vec![0i32; cell_count];
            for ind in &self.individuals {
                let pos = ind.position % cell_count;
                occupancy[pos] = occupancy[pos].saturating_add(1);
            }

            self.tensor
                .sync_from_individuals(&self.individuals, &self.config);
            let (decisions, gpu_ran) = run_brain_forward(
                &mut self.tensor,
                &mut self.individuals,
                &self.env.resources,
                &occupancy,
                &self.config,
                &mut self.nn_hidden_buf,
                &mut self.nn_out2_buf,
                &mut self.gpu_state,
                self.hex_dish.as_deref(),
            );
            let outcome = apply_decisions_sequential(
                &mut self.individuals,
                &decisions,
                &self.config,
                &mut self.env,
                self.hex_dish.as_deref(),
            );
            if gpu_ran {
                self.gpu_steps_executed = self.gpu_steps_executed.saturating_add(1);
                self.tensor
                    .apply_spatial_runtime_to_individuals(&mut self.individuals);
            } else {
                self.cpu_brain_steps_executed = self.cpu_brain_steps_executed.saturating_add(1);
            }
            self.last_spend = outcome.last_spend;
            self.last_delta = outcome.last_delta;
            self.last_movers = outcome.last_movers;
        } else {
            self.cpu_brain_steps_executed = self.cpu_brain_steps_executed.saturating_add(1);
            let outcome = step_individuals_cpu(
                &mut self.individuals,
                &mut self.env,
                &self.config,
                &mut self.nn_hidden_buf,
                &mut self.nn_out2_buf,
                &mut self.rng,
                self.hex_dish.as_deref(),
            );
            self.last_spend = outcome.last_spend;
            self.last_delta = outcome.last_delta;
            self.last_movers = outcome.last_movers;
        }

        for ind in &mut self.individuals {
            ind.age += 1;
        }

        let births = run_reproduction_phase(
            &mut self.individuals,
            &self.config,
            &mut self.rng,
            &mut self.next_id,
            |_| {},
        );
        if let Some(ref mut clado) = self.cladogram {
            record_births_in_cladogram(clado, &births);
        }
        record_births_in_lineage(&mut self.architecture_lineage, self.step_count, &births);

        let dead_ids = apply_metabolism_and_cull(&mut self.individuals, &self.config, &mut self.rng);
        if let Some(ref mut clado) = self.cladogram {
            for id in dead_ids {
                clado.mark_death(id);
            }
        }
        if self.use_gpu_step() {
            self.tensor
                .sync_from_individuals(&self.individuals, &self.config);
        }
    }

    pub fn step(&mut self) {
        if let Some(ref dish) = self.hex_dish {
            self.env.step_hex_dish(
                dish,
                self.config.seed as u64,
                self.config.env_time_speed,
                self.config.env_time_amplitude,
                self.config.env_perlin_spatial_scale,
                self.config.env_perlin_time_scale,
                self.config.resource_scale,
                self.config.resource_contrast,
            );
        } else {
            self.env.step(
                self.config.seed as u64,
                self.config.env_time_speed,
                self.config.env_time_amplitude,
                self.config.env_perlin_spatial_scale,
                self.config.env_perlin_time_scale,
                self.config.resource_scale,
                self.config.resource_contrast,
            );
        }
        self.step_with_existing_resources();
    }

    pub fn run(&mut self) {
        self.run_with_context(&RunRecordContext::default());
    }

    pub fn run_with_context(&mut self, ctx: &RunRecordContext) {
        use std::fs;
        use std::io::{self, Write};

        fn find_repo_root(mut cur: std::path::PathBuf) -> Option<std::path::PathBuf> {
            // Prefer the top-level project root that contains CMakeLists.txt; if none
            // is found, fall back to the highest directory with a README.md.
            let mut readme_candidate: Option<std::path::PathBuf> = None;
            loop {
                if cur.join("CMakeLists.txt").exists() {
                    return Some(cur);
                }
                if readme_candidate.is_none() && cur.join("README.md").exists() {
                    readme_candidate = Some(cur.clone());
                }
                if let Some(parent) = cur.parent() {
                    cur = parent.to_path_buf();
                } else {
                    return readme_candidate.or(Some(cur));
                }
            }
        }

        let steps = self.config.steps;
        let report_interval = if self.config.report_interval > 0 {
            self.config.report_interval
        } else {
            1
        };
        let progress_interval = progress_update_interval(steps, report_interval);
        let progress_tty = std::io::stdout().is_terminal();
        let write_frames = self.config.write_frames;
        let wants_frames = write_frames && self.config.write_outputs;
        let estimated_bytes = self.estimate_output_bytes(wants_frames);
        if !self.enforce_output_budget(estimated_bytes) {
            eprintln!("[error] aborting run because projected outputs exceed configured budget");
            return;
        }
        let needs_simdir = wants_frames
            || self.config.write_outputs
            || self.config.write_config
            || self.config.write_snapshot;

        let mut simdir = String::new();
        let mut writer: Option<WebviewerWriter> = None;
        if needs_simdir {
            let repo_root = std::env::current_dir()
                .ok()
                .and_then(find_repo_root)
                .unwrap_or_else(|| {
                    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
                });

            // If an explicit output directory is provided, write everything there.
            // Otherwise create/find a per-sim directory under `output_dir_base`.
            let simpath = if !self.config.output_run_dir.is_empty() {
                let p = std::path::Path::new(&self.config.output_run_dir);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    repo_root.join(p)
                }
            } else {
                let base = if !self.config.output_dir_base.is_empty() {
                    let wb = std::path::Path::new(&self.config.output_dir_base);
                    if wb.is_absolute() {
                        wb.to_path_buf()
                    } else {
                        repo_root.join(wb)
                    }
                } else {
                    repo_root.clone()
                };
                let _ = fs::create_dir_all(&base);

                let mut simnum = self.config.sim_number;
                if simnum < 0 {
                    let mut maxn = -1;
                    if let Ok(entries) = fs::read_dir(&base) {
                        for entry in entries.flatten() {
                            if let Ok(fname) = entry.file_name().into_string() {
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
                base.join(&dn)
            };

            let _ = fs::create_dir_all(&simpath);
            simdir = simpath.to_string_lossy().to_string();

            if self.config.write_outputs || wants_frames {
                let mut w = WebviewerWriter::new(&self.config, &simdir, self.ind_encoding);
                w.open_frames_if_needed();
                writer = Some(w);
            }

            if self.config.write_config {
                let cfg_json =
                    serde_json::to_string(&self.config).unwrap_or_else(|_| "{}".to_string());
                if let Some(w) = writer.as_mut() {
                    w.write_frames_config_header(&cfg_json);
                }
                self.write_config_artifacts(&simdir, &cfg_json);
            }

            if let Some(w) = writer.as_mut() {
                w.write_frames_header_row();
                if self.config.write_outputs {
                    w.dump_env(&self.env, Some(0));
                    w.dump_individuals(
                        &self.individuals,
                        &self.last_spend,
                        &self.last_delta,
                        Some(0),
                    );
                }
            }
        } else if self.config.write_outputs {
            // No simdir was created, but outputs were requested; fall back to a temporary
            // directory derived from output_dir_base.
            let temp_dir = if !self.config.output_dir_base.is_empty() {
                self.config.output_dir_base.clone()
            } else {
                "output".to_string()
            };
            let mut w = WebviewerWriter::new(&self.config, &temp_dir, self.ind_encoding);
            w.open_frames_if_needed();
            w.write_frames_header_row();
            w.dump_env(&self.env, Some(0));
            w.dump_individuals(
                &self.individuals,
                &self.last_spend,
                &self.last_delta,
                Some(0),
            );
            writer = Some(w);
        }

        let mut final_step_index = 0;
        let start_time = Instant::now();
        for t in 0..steps {
            final_step_index = t;
            self.step();

            if self.individuals.is_empty() {
                println!("[info] population extinct at t={}, stopping simulation", t);
                break;
            }

            if self.config.show_progress && should_emit_progress(t, steps, progress_interval) {
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
                let elapsed = start_time.elapsed().as_secs_f64();
                let avg = elapsed / ((t + 1) as f64);
                let rem = ((steps - (t + 1)) as f64) * avg;
                let hrs = (rem as i64) / 3600;
                let mins = ((rem as i64) % 3600) / 60;
                let secs = (rem as i64) % 60;
                ss.push_str(&format!("  ETA {}:{:02}:{:02}", hrs, mins, secs));

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

                if progress_tty {
                    ss.push('\r');
                    print!("{}", ss);
                } else {
                    println!("[progress] {}", ss.trim());
                }
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

                let frame = self.get_visualization(self.config.vis_cols, self.config.vis_height);
                if let Some(w) = writer.as_mut() {
                    w.write_frame_row(t, self.individuals.len(), avgres, &frame, prop_moved);
                }

                if let Some(w) = writer.as_mut() {
                    if self.config.write_outputs {
                        w.dump_env(&self.env, Some(t));
                        w.dump_individuals(
                            &self.individuals,
                            &self.last_spend,
                            &self.last_delta,
                            Some(t),
                        );
                    }
                    w.flush_output_buffers(t);
                    w.maybe_flush_nn_index(t);
                }
            }

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

        if self.config.show_progress {
            println!();
        }
        if let Some(w) = writer.as_mut() {
            if self.config.write_outputs {
                w.dump_env(&self.env, Some(final_step_index));
                w.dump_individuals(
                    &self.individuals,
                    &self.last_spend,
                    &self.last_delta,
                    Some(final_step_index),
                );
            }
            w.flush_output_buffers(final_step_index);
            w.flush_all();
        }

        if self.config.write_snapshot {
            self.persist_population_snapshot(&simdir, final_step_index, ctx);

            // After successfully writing the snapshot, copy the helper PowerShell
            // script into the simulation directory so that downstream processing
            // (e.g. generating webviewer outputs) is available next to the
            // produced artifacts by default.
            if !simdir.is_empty() {
                if let Ok(cwd) = std::env::current_dir() {
                    if let Some(root) = find_repo_root(cwd) {
                        let helper = root.join("scripts").join("generate_webviewer_outputs.ps1");
                        if helper.exists() {
                            let dst = std::path::Path::new(&simdir)
                                .join("generate_webviewer_outputs.ps1");
                            let _ = std::fs::copy(&helper, &dst);
                        }
                    }
                }
            }
        }

        // Write build id into the simulation directory for run compatibility tracking
        if !simdir.is_empty() {
            let mut bid_path = std::path::Path::new(&simdir).to_path_buf();
            bid_path.push("build_id.txt");
            let _ = std::fs::write(&bid_path, crate::build_info::VERSION_LABEL);
        }

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
        summary_map.insert(
            "final_population".to_string(),
            serde_json::Value::from(final_pop),
        );
        summary_map.insert(
            "prop_moved".to_string(),
            serde_json::Value::from(prop_moved),
        );
        if let Ok(s) = serde_json::to_string(&summary_map) {
            // include build id in the JSON summary so external drivers can record it
            let mut v: serde_json::Value =
                serde_json::from_str(&s).unwrap_or(serde_json::Value::Null);
            if let serde_json::Value::Object(ref mut map) = v {
                map.insert(
                    "version".to_string(),
                    serde_json::Value::from(crate::build_info::CRATE_VERSION),
                );
                map.insert(
                    "build_id".to_string(),
                    serde_json::Value::from(crate::build_info::BUILD_ID),
                );
            }
            if let Ok(s2) = serde_json::to_string(&v) {
                println!("{}", s2);
            } else {
                println!("{}", s);
            }
        }

        let steps_completed = final_step_index + 1;
        persist_run_record(
            self,
            ctx,
            &simdir,
            steps,
            steps_completed,
            start_time.elapsed().as_secs_f64(),
            final_pop,
            prop_moved,
        );
    }

    fn write_config_artifacts(&self, simdir: &str, config_json: &str) {
        if simdir.is_empty() {
            return;
        }
        let simpath = std::path::Path::new(simdir);
        let mut cfg_tmp = simpath.to_path_buf();
        cfg_tmp.push("config.json.tmp");
        if let Ok(mut cout) = std::fs::File::create(&cfg_tmp) {
            let _ = write!(cout, "{}", config_json);
            let _ = cout.flush();
            let mut finalp = simpath.to_path_buf();
            finalp.push("config.json");
            let _ = std::fs::rename(&cfg_tmp, &finalp);
        }

        // Convenience config for postprocessing / webviewer generation.
        // This is a copy of config.json with a shorter run and outputs enabled.
        // It is intended for generating visualization/experiment artifacts without
        // re-writing snapshots or configs into run directories.
        let run_config_json = (|| {
            let mut v: serde_json::Value = serde_json::from_str(config_json).ok()?;
            if let serde_json::Value::Object(map) = &mut v {
                map.insert("burn_in".to_string(), serde_json::Value::from(0));
                map.insert("steps".to_string(), serde_json::Value::from(100));
                map.insert("write_outputs".to_string(), serde_json::Value::from(true));
                map.insert("write_snapshot".to_string(), serde_json::Value::from(false));
                map.insert("write_config".to_string(), serde_json::Value::from(false));
                map.insert("write_frames".to_string(), serde_json::Value::from(false));
            }
            serde_json::to_string_pretty(&v).ok()
        })()
        .unwrap_or_else(|| config_json.to_string());

        let mut run_cfg_tmp = simpath.to_path_buf();
        run_cfg_tmp.push("run_config.json.tmp");
        if let Ok(mut cout) = std::fs::File::create(&run_cfg_tmp) {
            let _ = write!(cout, "{}", run_config_json);
            let _ = cout.flush();
            let mut finalp = simpath.to_path_buf();
            finalp.push("run_config.json");
            let _ = std::fs::rename(&run_cfg_tmp, &finalp);
        }

        if !self.config.config_out_path.is_empty() {
            let custom = simpath.join(&self.config.config_out_path);
            if let Some(parent) = custom.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut custom_tmp = custom.clone();
            if let Some(fname) = custom.file_name() {
                let tmpname = format!("{}.tmp", fname.to_string_lossy());
                custom_tmp.set_file_name(tmpname);
            } else {
                custom_tmp.push(".tmp");
            }
            if let Ok(mut coutf) = std::fs::File::create(&custom_tmp) {
                let _ = write!(coutf, "{}", config_json);
                let _ = coutf.flush();
                let _ = std::fs::rename(&custom_tmp, &custom);
            }
        }
    }

    fn persist_population_snapshot(
        &self,
        simdir: &str,
        timestep: i32,
        ctx: &RunRecordContext,
    ) {
        if !self.config.write_snapshot {
            return;
        }
        if simdir.is_empty() {
            eprintln!(
                "[warn] snapshot requested but no simulation directory was created; skipping"
            );
            return;
        }
        let env_height = if self.config.hex_dish_radius > 0 {
            self.hex_dish
                .as_ref()
                .map(|d| d.bounding_size as u32)
                .unwrap_or(0)
        } else {
            self.config.grid_height.max(self.config.grid_width) as u32
        };
        let hex_dish_radius = if self.config.hex_dish_radius > 0 {
            self.config.hex_dish_radius.max(0) as u32
        } else {
            0
        };
        let snapshot = PopulationSnapshot::capture(
            &self.env,
            &self.individuals,
            self.next_id,
            self.config.seed as u64,
            timestep,
            env_height,
            hex_dish_radius,
        );
        let mut path = std::path::PathBuf::from(simdir);
        let fname = if self.config.snapshot_filename.is_empty() {
            "population_snapshot.cbor"
        } else {
            &self.config.snapshot_filename
        };
        path.push(fname);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(err) = snapshot.write_to_path(&path) {
            match err {
                SnapshotError::UnsupportedVersion { expected, found } => {
                    eprintln!(
                        "[warn] failed to write snapshot (version {} vs expected {}): {}",
                        found, expected, err
                    );
                }
                _ => {
                    eprintln!(
                        "[warn] failed to write snapshot to {}: {}",
                        path.display(),
                        err
                    );
                }
            }
        } else if path.is_file() {
            snapshot_registry::register_headless_snapshot(
                &snapshot,
                &path,
                simdir,
                &self.config,
                ctx,
                self.individuals.len(),
                None,
            );
        }

        if self.config.export_cladogram {
            if let Some(ref clado) = self.cladogram {
                let clado_path = std::path::Path::new(simdir).join("cladogram.json");
                if let Err(e) = clado.write_json(&clado_path) {
                    eprintln!("[warn] failed to write cladogram: {}", e);
                }
            }
            let lineage_path = std::path::Path::new(simdir).join("architecture_lineage.jsonl");
            if let Err(e) = write_lineage_jsonl(&self.architecture_lineage, &lineage_path) {
                eprintln!("[warn] failed to write architecture lineage: {}", e);
            }
        }
    }

    /// Load a previously captured population snapshot from disk and replace the
    /// current environment/individuals state. Used for replay runs.
    pub fn load_snapshot_from_path<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), SnapshotError> {
        let snapshot = PopulationSnapshot::read_from_path(&path)?;

        // Restore environment dimensions and time.
        let width = snapshot.env_width as usize;
        self.env.width = width;
        self.env.perlin_time = snapshot.env_perlin_time;

        if self.config.replay_env_seed >= 0 {
            // When an override seed is provided, rebuild the Perlin base using that seed
            // so we can test the same population under a different landscape.
            let seed = self.config.replay_env_seed as u64;
            self.env = Environment::new(width);
            self.env.init_perlin(
                seed,
                self.config.resource_scale,
                self.config.resource_contrast,
            );
            self.env.perlin_time = snapshot.env_perlin_time;
        } else {
            // Otherwise, restore the exact environment from the snapshot.
            self.env.resources = snapshot.env_resources.iter().map(|v| *v as f64).collect();
            self.env.perlin_base = snapshot.env_perlin_base.iter().map(|v| *v as f64).collect();
        }

        // Rebuild individuals from snapshot records.
        self.individuals = snapshot
            .individuals
            .into_iter()
            .map(|rec| rec.into_individual())
            .collect();

        // Optionally randomize positions on replay to test robustness under reshuffling.
        if self.config.replay_randomize_start_positions {
            use rand::distributions::{Distribution, Uniform};
            let dist = Uniform::new(0usize, width.max(1));
            for ind in &mut self.individuals {
                ind.position = dist.sample(&mut self.rng);
            }
        }

        // Restore next_id so new individuals get unique IDs.
        self.next_id = snapshot.next_id;

        // Reset book-keeping buffers to match the restored population.
        self.last_spend.clear();
        self.last_delta.clear();
        self.last_movers = 0;

        Ok(())
    }

    fn estimate_output_bytes(&self, frames_enabled: bool) -> u64 {
        let mut total: u64 = 0;
        let history_steps = self.estimate_history_steps() as u64;
        if self.config.write_outputs {
            let env_record = (self.env.width.max(1) * 16 + 128) as u64;
            total = total.saturating_add(env_record.saturating_mul(history_steps));

            let pop_ceiling = self.estimated_population_ceiling() as u64;
            if self.config.write_individual_deltas {
                let per_entry = self.estimate_individual_delta_entry_bytes();
                let (full_frames, delta_frames) = self.estimate_delta_frame_counts(history_steps);
                let updates_per_frame =
                    self.estimate_delta_updates_per_frame(pop_ceiling as usize) as u64;
                let removals_per_frame =
                    self.estimate_delta_removals_per_frame(pop_ceiling as usize) as u64;

                let full_record = 96u64.saturating_add(pop_ceiling.saturating_mul(per_entry));
                total = total.saturating_add(full_record.saturating_mul(full_frames));

                let delta_record = 96u64
                    .saturating_add(updates_per_frame.saturating_mul(per_entry))
                    .saturating_add(removals_per_frame.saturating_mul(8));
                total = total.saturating_add(delta_record.saturating_mul(delta_frames));
            } else {
                let per_individual = self.estimate_individual_entry_bytes();
                let ind_record = 96u64.saturating_add(pop_ceiling.saturating_mul(per_individual));
                total = total.saturating_add(ind_record.saturating_mul(history_steps));
            }
        }

        if frames_enabled {
            let frame_chars = (self.config.vis_cols.max(1)
                * (self.config.vis_height + self.config.vis_pop_height + 6))
                as u64;
            total = total.saturating_add((frame_chars + 64).saturating_mul(history_steps));
        }

        if self.config.write_snapshot {
            let env_blob = (self.env.width.max(1) * 16 * 2) as u64;
            let pop_ceiling = self.estimated_population_ceiling() as u64;
            let per_snap = self.estimate_snapshot_entry_bytes();
            total = total.saturating_add(env_blob);
            total = total.saturating_add(pop_ceiling.saturating_mul(per_snap));
        }

        total
    }

    fn estimate_individual_entry_bytes(&self) -> u64 {
        match self.ind_encoding {
            IndividualsEncoding::JsonPretty => 96,
            IndividualsEncoding::JsonCompact => 72,
            IndividualsEncoding::Cbor => 48,
        }
    }

    fn estimate_individual_delta_entry_bytes(&self) -> u64 {
        // Delta frames carry the same fields as compact individual records (id, pos,
        // resources, last_spend, last_delta, age) without any NN weights.
        match self.ind_encoding {
            IndividualsEncoding::JsonPretty => 80,
            IndividualsEncoding::JsonCompact => 64,
            IndividualsEncoding::Cbor => 40,
        }
    }

    fn estimate_snapshot_entry_bytes(&self) -> u64 {
        let nodes = self.config.nn_nodes.max(0) as usize;
        let nodes2 = self.config.nn_nodes2.max(0) as usize;
        let inputs = self.config.nn_inputs.max(0) as usize;
        let mut bytes = 32usize;
        bytes += nodes * 4;
        bytes += nodes2 * 4;
        bytes += nodes * inputs * 4;
        bytes += nodes * nodes2 * 4;
        bytes as u64
    }

    fn estimate_history_steps(&self) -> usize {
        let steps = self.config.steps.max(1) as i64;
        let burn = self.config.burn_in.max(0) as i64;
        let interval = self.config.report_interval.max(1) as i64;
        let active = (steps - burn).max(0);
        let mut frames = active / interval;
        if active % interval != 0 {
            frames += 1;
        }
        frames as usize + 2
    }

    fn estimated_population_ceiling(&self) -> usize {
        let current = self.individuals.len().max(1);
        let initial = self.config.initial_population.max(0) as usize;
        let base = current.max(initial);
        let headroom = ((base as f64) * 1.5).ceil() as usize + 128;
        let with_headroom = if self.config.max_population > 0 {
            headroom.min(self.config.max_population as usize)
        } else {
            headroom
        };
        with_headroom.max(base)
    }

    fn estimate_delta_frame_counts(&self, history_steps: u64) -> (u64, u64) {
        if history_steps == 0 {
            return (0, 0);
        }
        let full_every = self.config.delta_full_interval.max(1) as u64;
        let full_frames = 1u64 + history_steps.saturating_sub(1) / full_every;
        let delta_frames = history_steps.saturating_sub(full_frames);
        (full_frames, delta_frames)
    }

    fn estimate_delta_updates_per_frame(&self, pop_ceiling: usize) -> usize {
        if pop_ceiling == 0 {
            return 0;
        }
        if !self.last_delta.is_empty() && !self.individuals.is_empty() {
            let seen = std::cmp::min(self.last_delta.len(), self.individuals.len());
            if seen > 0 {
                let moved = self
                    .last_delta
                    .iter()
                    .take(seen)
                    .filter(|&&d| d != 0)
                    .count();
                let frac = (moved as f64 / seen as f64).clamp(0.0, 1.0);
                let est = (frac * pop_ceiling as f64).ceil() as usize;
                return est.max(seen);
            }
        }
        // Fallback heuristic: assume around half the population produces updates per frame.
        ((pop_ceiling as f64) * 0.5).ceil() as usize
    }

    fn estimate_delta_removals_per_frame(&self, pop_ceiling: usize) -> usize {
        if pop_ceiling == 0 {
            return 0;
        }
        ((pop_ceiling as f64) * 0.02).ceil() as usize
    }

    fn enforce_output_budget(&self, estimated_bytes: u64) -> bool {
        let est_mb = estimated_bytes as f64 / (1024.0 * 1024.0);
        if self.config.output_budget_mb <= 0.0 {
            if estimated_bytes > 0 {
                println!(
                    "[info] projected outputs ~{:.2} MiB (no budget configured)",
                    est_mb
                );
            }
            return true;
        }

        let budget_bytes = (self.config.output_budget_mb.max(0.0) * 1024.0 * 1024.0) as u64;
        if budget_bytes == 0 {
            return true;
        }
        let warn_fraction = self.config.output_budget_warn_fraction.clamp(0.0, 1.0);
        let warn_bytes = (budget_bytes as f64 * warn_fraction) as u64;
        let budget_mb = budget_bytes as f64 / (1024.0 * 1024.0);

        if estimated_bytes >= budget_bytes {
            eprintln!(
                "[warn] projected outputs {:.2} MiB exceed configured budget {:.2} MiB",
                est_mb, budget_mb
            );
            if self.config.output_budget_hard_stop {
                return false;
            }
        } else if warn_bytes > 0 && estimated_bytes >= warn_bytes {
            let pct = (estimated_bytes as f64 / budget_bytes as f64) * 100.0;
            eprintln!(
                "[warn] projected outputs {:.2} MiB reach {:.1}% of budget {:.2} MiB",
                est_mb, pct, budget_mb
            );
        } else {
            println!(
                "[info] projected outputs {:.2} MiB within budget {:.2} MiB",
                est_mb, budget_mb
            );
        }

        true
    }

    fn resolve_individuals_encoding(mode: &str) -> IndividualsEncoding {
        resolve_individuals_encoding(mode)
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
                "    brain={:?} complexity={}\n",
                ind.brain.kind(),
                ind.complexity()
            ));
        }
        s
    }

    /// Print the debug snapshot to stdout.
    pub fn print_debug_snapshot(&self, sample_count: usize) {
        println!("{}", self.debug_snapshot(sample_count));
    }
}

#[cfg(test)]
mod progress_tests {
    use super::{progress_update_interval, should_emit_progress};

    #[test]
    fn progress_interval_throttles_long_runs() {
        assert_eq!(progress_update_interval(100_000, 2000), 2000);
        assert_eq!(progress_update_interval(100_000, 1), 2000);
    }

    #[test]
    fn progress_emits_at_start_and_end() {
        let interval = 2000;
        assert!(should_emit_progress(0, 100_000, interval));
        assert!(should_emit_progress(99_999, 100_000, interval));
        assert!(!should_emit_progress(1, 100_000, interval));
        assert!(should_emit_progress(2000, 100_000, interval));
    }
}
