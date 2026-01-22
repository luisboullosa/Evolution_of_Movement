use crate::config::Config;
use crate::environment::Environment;
use crate::individual::Individual;
use crate::neural_network::NNType;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndividualsEncoding {
    JsonPretty,
    JsonCompact,
    Cbor,
}

#[derive(Serialize)]
struct CompactIndividual {
    i: i32,
    p: u32,
    r: f32,
    s: f32,
    d: i32,
    a: i32,
}

#[derive(Serialize)]
struct CompactIndividualsFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    t: Option<i32>,
    #[serde(rename = "ind")]
    ind: Vec<CompactIndividual>,
}

pub(crate) fn resolve_individuals_encoding(mode: &str) -> IndividualsEncoding {
    let normalized = mode.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "jsonl" | "json" | "pretty" | "ndjson" => IndividualsEncoding::JsonPretty,
        "jsonl-compact" | "jsonl_compact" | "compact" | "json-compact" | "compact-json" => {
            IndividualsEncoding::JsonCompact
        }
        "cbor" => IndividualsEncoding::Cbor,
        _ => IndividualsEncoding::JsonPretty,
    }
}

fn ensure_nn_index_entry(cache: &mut HashMap<String, String>, ind: &Individual) -> bool {
    use std::collections::hash_map::Entry;
    match cache.entry(ind.id.to_string()) {
        Entry::Occupied(_) => false,
        Entry::Vacant(e) => {
            let mut js = String::new();
            js.push('{');
            let ntype = match ind.nn.nn_type {
                NNType::SingleInput => "single",
                _ => "multi",
            };
            js.push_str(&format!("\n      \"type\": \"{}\",\n", ntype));
            js.push_str(&format!(
                "      \"inputs\": {},\n      \"nodes\": {},\n      \"layers\": {},\n      \"nodes2\": {}\n",
                ind.nn.inputs, ind.nn.nodes, ind.nn.layers, ind.nn.nodes2
            ));
            js.push_str("      ,\"weights\": ");
            if let NNType::SingleInput = ind.nn.nn_type {
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
                    for j in 0..ind
                        .nn
                        .inputs
                        .min(ind.nn.weights_multi.len().saturating_div(ind.nn.nodes))
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
            js.push_str(",      \"biases\": [");
            for (bi, b) in ind.nn.biases.iter().enumerate() {
                if bi > 0 {
                    js.push_str(", ");
                }
                js.push_str(&format!("{:.6}", b));
            }
            js.push_str("]\n");
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
            true
        }
    }
}

pub(crate) struct WebviewerWriter {
    cfg: Config,
    simdir: PathBuf,
    ind_encoding: IndividualsEncoding,

    frames_file: Option<File>,
    env_file: Option<File>,
    ind_file: Option<File>,

    delta_file: Option<File>,
    prev_snapshot: HashMap<i32, (usize, f64, i32, f64, i32)>, // id -> (pos, resources, age, last_spend, last_delta)
    frame_count: i32,

    nn_index_cache: HashMap<String, String>,
    nn_index_dirty: bool,
}

impl WebviewerWriter {
    pub(crate) fn new(cfg: &Config, simdir: &str, ind_encoding: IndividualsEncoding) -> Self {
        WebviewerWriter {
            cfg: cfg.clone(),
            simdir: PathBuf::from(simdir),
            ind_encoding,
            frames_file: None,
            env_file: None,
            ind_file: None,
            delta_file: None,
            prev_snapshot: HashMap::new(),
            frame_count: 0,
            nn_index_cache: HashMap::new(),
            nn_index_dirty: false,
        }
    }

    pub(crate) fn wants_frames(&self) -> bool {
        self.cfg.write_frames && self.cfg.write_outputs
    }

    pub(crate) fn open_frames_if_needed(&mut self) {
        if !self.wants_frames() {
            return;
        }
        if self.frames_file.is_some() {
            return;
        }
        let fpath = self.simdir.join(&self.cfg.frames_filename);
        if let Some(parent) = fpath.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(f) = std::fs::File::create(&fpath) {
            self.frames_file = Some(f);
        }
    }

    pub(crate) fn write_frames_config_header(&mut self, cfg_json: &str) {
        if let Some(f) = self.frames_file.as_mut() {
            let _ = writeln!(f, "#CONFIG {}", cfg_json);
        }
    }

    pub(crate) fn write_frames_header_row(&mut self) {
        if let Some(f) = self.frames_file.as_mut() {
            let _ = writeln!(f, "t\tpop\tavg_res\tframe\tprop_moved");
        }
    }

    pub(crate) fn write_frame_row(
        &mut self,
        t: i32,
        pop: usize,
        avg_res: f64,
        frame: &str,
        prop_moved: f64,
    ) {
        if let Some(f) = self.frames_file.as_mut() {
            let mut esc = String::new();
            for ch in frame.chars() {
                if ch == '\n' {
                    esc.push_str("\\n");
                } else {
                    esc.push(ch);
                }
            }
            let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}", t, pop, avg_res, esc, prop_moved);
        }
    }

    fn env_path(&self) -> PathBuf {
        let env_fname = Path::new(&self.cfg.env_out_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("env.jsonl");
        self.simdir.join(env_fname)
    }

    fn individuals_path(&self) -> PathBuf {
        let fname = if !self.cfg.individuals_out_path.is_empty() {
            Path::new(&self.cfg.individuals_out_path)
                .file_name()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        } else {
            None
        };
        let fname = fname.unwrap_or_else(|| self.default_individuals_filename().to_string());
        self.simdir.join(fname)
    }

    fn default_individuals_filename(&self) -> &'static str {
        match self.ind_encoding {
            IndividualsEncoding::JsonPretty | IndividualsEncoding::JsonCompact => {
                "individuals.jsonl"
            }
            IndividualsEncoding::Cbor => "individuals.cbor",
        }
    }

    fn open_env_if_needed(&mut self) {
        if self.env_file.is_some() {
            return;
        }
        let path = self.env_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
            self.env_file = Some(f);
        }
    }

    fn open_individuals_if_needed(&mut self) {
        if self.ind_file.is_some() {
            return;
        }
        let path = self.individuals_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
            self.ind_file = Some(f);
        }
    }

    fn delta_path(&self) -> PathBuf {
        let fname = if !self.cfg.individuals_delta_out_path.is_empty() {
            Path::new(&self.cfg.individuals_delta_out_path)
                .file_name()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        } else {
            None
        };
        let fname = fname.unwrap_or_else(|| "individuals.delta.jsonl".to_string());
        self.simdir.join(fname)
    }

    fn open_delta_if_needed(&mut self) {
        if self.delta_file.is_some() {
            return;
        }
        let path = self.delta_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
            self.delta_file = Some(f);
        }
    }

    pub(crate) fn dump_env(&mut self, env: &Environment, t: Option<i32>) {
        if !self.cfg.write_outputs {
            return;
        }
        self.open_env_if_needed();
        if let Some(f) = self.env_file.as_mut() {
            let _ = write!(f, "{{");
            if let Some(tv) = t {
                let _ = write!(f, "\"t\":{}, ", tv);
            }
            let _ = write!(f, "\"resources\": [");
            for i in 0..env.width {
                if i > 0 {
                    let _ = write!(f, ", ");
                }
                let _ = write!(f, "{:.6}", env.resources[i]);
            }
            let _ = writeln!(f, "]}}");
            let _ = f.flush();
        }
    }

    pub(crate) fn dump_individuals(
        &mut self,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) {
        if !self.cfg.write_outputs {
            return;
        }
        if self.cfg.write_individual_deltas {
            self.dump_individuals_delta(individuals, last_spend, last_delta, t);
        } else {
            self.open_individuals_if_needed();
            if let Some(mut file) = self.ind_file.take() {
                let _ = self.write_individuals_record(&mut file, individuals, last_spend, last_delta, t);
                let _ = file.flush();
                self.ind_file = Some(file);
            }
        }
    }

    fn dump_individuals_delta(
        &mut self,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) {
        self.open_delta_if_needed();

        let width = self.cfg.grid_width.max(1);
        let mut seen: HashMap<i32, (usize, f64, i32, f64, i32)> = HashMap::new();
        for (i, ind) in individuals.iter().enumerate() {
            let spend = last_spend.get(i).copied().unwrap_or(0.0);
            let delta = last_delta.get(i).copied().unwrap_or(0);
            seen.insert(ind.id, (ind.position % width, ind.resources, ind.age, spend, delta));
            if ensure_nn_index_entry(&mut self.nn_index_cache, ind) {
                self.nn_index_dirty = true;
            }
        }

        let need_full = self.prev_snapshot.is_empty()
            || (self.cfg.delta_full_interval > 0
                && (self.frame_count % self.cfg.delta_full_interval) == 0);

        if need_full {
            if let Some(mut file) = self.delta_file.take() {
                let _ = write!(file, "{{");
                if let Some(tv) = t {
                    let _ = write!(file, "\"t\":{}, ", tv);
                }
                let _ = write!(file, "\"full\":true, \"individuals\":[");
                let mut first = true;
                for (id, (pos, res, age, sp, de)) in seen.iter() {
                    if !first {
                        let _ = write!(file, ", ");
                    }
                    first = false;
                    let _ = write!(
                        file,
                        "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"age\":{},\"last_spend\":{:.6},\"last_delta\":{}}}",
                        id, pos, res, age, sp, de
                    );
                }
                let _ = writeln!(file, "]}}");
                let _ = file.flush();
                self.delta_file = Some(file);
            }
            self.prev_snapshot = seen;
            self.frame_count += 1;
            return;
        }

        let mut updates: Vec<String> = Vec::new();
        let mut removals: Vec<i32> = Vec::new();

        for (id, (pos, res, age, sp, de)) in seen.iter() {
            match self.prev_snapshot.get(id) {
                None => {
                    updates.push(format!(
                        "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"age\":{},\"last_spend\":{:.6},\"last_delta\":{}}}",
                        id, pos, res, age, sp, de
                    ));
                }
                Some((p_pos, p_res, p_age, p_sp, p_de)) => {
                    if *p_pos != *pos || (*p_de != *de) || (*p_sp != *sp) || (*p_res != *res) || (*p_age != *age) {
                        updates.push(format!(
                            "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"age\":{},\"last_spend\":{:.6},\"last_delta\":{}}}",
                            id, pos, res, age, sp, de
                        ));
                    }
                }
            }
        }

        for id in self.prev_snapshot.keys() {
            if !seen.contains_key(id) {
                removals.push(*id);
            }
        }

        if updates.is_empty() && removals.is_empty() {
            self.prev_snapshot = seen;
            self.frame_count += 1;
            return;
        }

        if let Some(mut file) = self.delta_file.take() {
            let _ = write!(file, "{{");
            if let Some(tv) = t {
                let _ = write!(file, "\"t\":{}, ", tv);
            }
            let _ = write!(file, "\"updates\":[");
            for (i, u) in updates.iter().enumerate() {
                if i > 0 {
                    let _ = write!(file, ",");
                }
                let _ = write!(file, "{}", u);
            }
            let _ = write!(file, "], \"remove\":[");
            for (i, rid) in removals.iter().enumerate() {
                if i > 0 {
                    let _ = write!(file, ",");
                }
                let _ = write!(file, "{}", rid);
            }
            let _ = writeln!(file, "]}}");
            let _ = file.flush();
            self.delta_file = Some(file);
        }

        self.prev_snapshot = seen;
        self.frame_count += 1;
    }

    fn write_individuals_record<W: Write>(
        &mut self,
        writer: &mut W,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) -> std::io::Result<()> {
        match self.ind_encoding {
            IndividualsEncoding::JsonPretty => {
                self.write_individuals_json(writer, individuals, last_spend, last_delta, t)
            }
            IndividualsEncoding::JsonCompact => {
                self.write_individuals_compact_json(writer, individuals, last_spend, last_delta, t)
            }
            IndividualsEncoding::Cbor => {
                self.write_individuals_cbor(writer, individuals, last_spend, last_delta, t)
            }
        }
    }

    fn write_individuals_json<W: Write>(
        &mut self,
        writer: &mut W,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) -> std::io::Result<()> {
        write!(writer, "{{")?;
        if let Some(tv) = t {
            write!(writer, "\"t\":{}, ", tv)?;
        }
        write!(writer, "\"individuals\": [")?;
        for (i, ind) in individuals.iter().enumerate() {
            if i > 0 {
                write!(writer, ", ")?;
            }
            let spend = last_spend.get(i).copied().unwrap_or(0.0);
            let delta = last_delta.get(i).copied().unwrap_or(0);
            write!(
                writer,
                "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"last_spend\":{:.6},\"last_delta\":{},\"age\":{} }}",
                ind.id,
                ind.position,
                ind.resources,
                spend,
                delta,
                ind.age
            )?;
            if ensure_nn_index_entry(&mut self.nn_index_cache, ind) {
                self.nn_index_dirty = true;
            }
        }
        writeln!(writer, "]}}")?;
        Ok(())
    }

    fn write_individuals_compact_json<W: Write>(
        &mut self,
        writer: &mut W,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) -> std::io::Result<()> {
        let mut compact = Vec::with_capacity(individuals.len());
        for (i, ind) in individuals.iter().enumerate() {
            let spend = last_spend.get(i).copied().unwrap_or(0.0);
            let delta = last_delta.get(i).copied().unwrap_or(0);
            compact.push(CompactIndividual {
                i: ind.id,
                p: ind.position as u32,
                r: ind.resources as f32,
                s: spend as f32,
                d: delta,
                a: ind.age,
            });
            if ensure_nn_index_entry(&mut self.nn_index_cache, ind) {
                self.nn_index_dirty = true;
            }
        }
        let frame = CompactIndividualsFrame { t, ind: compact };
        serde_json::to_writer(&mut *writer, &frame)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writer.write_all(b"\n")
    }

    fn write_individuals_cbor<W: Write>(
        &mut self,
        writer: &mut W,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) -> std::io::Result<()> {
        let mut compact = Vec::with_capacity(individuals.len());
        for (i, ind) in individuals.iter().enumerate() {
            let spend = last_spend.get(i).copied().unwrap_or(0.0);
            let delta = last_delta.get(i).copied().unwrap_or(0);
            compact.push(CompactIndividual {
                i: ind.id,
                p: ind.position as u32,
                r: ind.resources as f32,
                s: spend as f32,
                d: delta,
                a: ind.age,
            });
            if ensure_nn_index_entry(&mut self.nn_index_cache, ind) {
                self.nn_index_dirty = true;
            }
        }
        let frame = CompactIndividualsFrame { t, ind: compact };
        serde_cbor::to_writer(&mut *writer, &frame)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub(crate) fn maybe_flush_nn_index(&mut self, t: i32) {
        if !self.cfg.write_outputs || !self.nn_index_dirty {
            return;
        }
        let do_flush = (self.cfg.nn_index_flush_interval > 0 && t % self.cfg.nn_index_flush_interval == 0)
            || (self.cfg.nn_index_flush_size > 0 && self.nn_index_cache.len() >= self.cfg.nn_index_flush_size);
        if !do_flush {
            return;
        }
        self.flush_nn_index();
    }

    pub(crate) fn flush_nn_index_final(&mut self) {
        if !self.cfg.write_outputs || !self.nn_index_dirty || self.nn_index_cache.is_empty() {
            return;
        }
        self.flush_nn_index();
    }

    fn flush_nn_index(&mut self) {
        let mut p = self.simdir.clone();
        let _ = std::fs::create_dir_all(&p);
        p.push("nn_index.json.tmp");
        if let Ok(mut jf) = std::fs::File::create(&p) {
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
            let mut finalp = self.simdir.clone();
            finalp.push("nn_index.json");
            let _ = std::fs::rename(&p, &finalp);
            self.nn_index_dirty = false;
        }
    }

    pub(crate) fn flush_all(&mut self) {
        if let Some(mut f) = self.frames_file.take() {
            let _ = f.flush();
        }
        self.flush_nn_index_final();
        if let Some(mut f) = self.env_file.take() {
            let _ = f.flush();
        }
        if let Some(mut f) = self.ind_file.take() {
            let _ = f.flush();
        }
        if let Some(mut f) = self.delta_file.take() {
            let _ = f.flush();
        }
    }
}
