use crate::config::Config;
use crate::environment::Environment;
use crate::individual::Individual;
use crate::neural_network::NNType;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
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

struct BufferWriter<'a> {
    buf: &'a mut Vec<u8>,
}

impl<'a> BufferWriter<'a> {
    fn new(buf: &'a mut Vec<u8>) -> Self {
        BufferWriter { buf }
    }
}

impl<'a> Write for BufferWriter<'a> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BufferSummary {
    frames_bytes: usize,
    env_bytes: usize,
    individuals_bytes: usize,
    delta_bytes: usize,
}

impl BufferSummary {
    fn total(&self) -> usize {
        self.frames_bytes + self.env_bytes + self.individuals_bytes + self.delta_bytes
    }
}

fn ensure_nn_index_entry(cache: &mut HashMap<String, String>, ind: &Individual) -> bool {
    use std::collections::hash_map::Entry;
    match cache.entry(ind.id.to_string()) {
        Entry::Occupied(_) => false,
        Entry::Vacant(e) => {
            let mut js = String::new();
            js.push('{');
            let ntype = match ind.nn().nn_type {
                NNType::SingleInput => "single",
                _ => "multi",
            };
            js.push_str(&format!("\n      \"type\": \"{}\",\n", ntype));
            js.push_str(&format!(
                "      \"inputs\": {},\n      \"nodes\": {},\n      \"layers\": {},\n      \"nodes2\": {}\n",
                ind.nn().inputs, ind.nn().nodes, ind.nn().layers, ind.nn().nodes2
            ));
            js.push_str("      ,\"weights\": ");
            if let NNType::SingleInput = ind.nn().nn_type {
                js.push('[');
                for (wi, w) in ind.nn().weights.iter().enumerate() {
                    if wi > 0 {
                        js.push_str(", ");
                    }
                    js.push_str(&format!("{:.6}", w));
                }
                js.push_str("]\n");
            } else {
                js.push('[');
                for ni in 0..ind.nn().nodes {
                    if ni > 0 {
                        js.push_str(", ");
                    }
                    js.push('[');
                    for j in 0..ind
                        .nn()
                        .inputs
                        .min(ind.nn().weights_multi.len().saturating_div(ind.nn().nodes))
                    {
                        if j > 0 {
                            js.push_str(", ");
                        }
                        let idx = ni.saturating_mul(ind.nn().inputs).saturating_add(j);
                        js.push_str(&format!(
                            "{:.6}",
                            ind.nn().weights_multi.get(idx).copied().unwrap_or(0.0)
                        ));
                    }
                    js.push(']');
                }
                js.push_str("]\n");
            }
            js.push_str(",      \"biases\": [");
            for (bi, b) in ind.nn().biases.iter().enumerate() {
                if bi > 0 {
                    js.push_str(", ");
                }
                js.push_str(&format!("{:.6}", b));
            }
            js.push_str("]\n");
            if ind.nn().layers == 2 {
                js.push_str(",      \"layer2_weights\": [");
                for ni in 0..ind.nn().nodes2 {
                    if ni > 0 {
                        js.push_str(", ");
                    }
                    js.push('[');
                    for j in 0..ind.nn().nodes {
                        if j > 0 {
                            js.push_str(", ");
                        }
                        let idx = ni.saturating_mul(ind.nn().nodes).saturating_add(j);
                        js.push_str(&format!(
                            "{:.6}",
                            ind.nn().weights2.get(idx).copied().unwrap_or(0.0)
                        ));
                    }
                    js.push(']');
                }
                js.push_str("]\n");
                js.push_str(",      \"layer2_biases\": [");
                for (bi, b) in ind.nn().biases2.iter().enumerate() {
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
    frames_buffer: Vec<u8>,
    env_buffer: Vec<u8>,
    individuals_buffer: Vec<u8>,
    delta_buffer: Vec<u8>,
    total_bytes_flushed: u64,
    max_buffer_bytes: usize,
    output_budget_bytes: Option<u64>,
    output_budget_warn_bytes: u64,
}

impl WebviewerWriter {
    pub(crate) fn new(cfg: &Config, simdir: &str, ind_encoding: IndividualsEncoding) -> Self {
        let budget_bytes = if cfg.output_budget_mb > 0.0 {
            Some((cfg.output_budget_mb * 1024.0 * 1024.0) as u64)
        } else {
            None
        };
        let warn_bytes = budget_bytes
            .map(|b| (b as f64 * cfg.output_budget_warn_fraction.clamp(0.0, 1.0)) as u64)
            .unwrap_or(0);
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
            frames_buffer: Vec::new(),
            env_buffer: Vec::new(),
            individuals_buffer: Vec::new(),
            delta_buffer: Vec::new(),
            total_bytes_flushed: 0,
            max_buffer_bytes: 0,
            output_budget_bytes: budget_bytes,
            output_budget_warn_bytes: warn_bytes,
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
        if !self.wants_frames() {
            return;
        }
        let mut esc = String::new();
        for ch in frame.chars() {
            if ch == '\n' {
                esc.push_str("\\n");
            } else {
                esc.push(ch);
            }
        }
        let row = format!("{}\t{}\t{}\t{}\t{}\n", t, pop, avg_res, esc, prop_moved);
        self.frames_buffer.extend_from_slice(row.as_bytes());
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
        let mut writer = BufferWriter::new(&mut self.env_buffer);
        let _ = write!(writer, "{{");
        if let Some(tv) = t {
            let _ = write!(writer, "\"t\":{}, ", tv);
        }
        let _ = write!(writer, "\"resources\": [");
        for i in 0..env.width {
            if i > 0 {
                let _ = write!(writer, ", ");
            }
            let _ = write!(writer, "{:.6}", env.resources[i]);
        }
        let _ = writeln!(writer, "]}}");
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
            let mut tmp = Vec::new();
            {
                let mut writer = BufferWriter::new(&mut tmp);
                let _ = self.write_individuals_record(
                    &mut writer,
                    individuals,
                    last_spend,
                    last_delta,
                    t,
                );
            }
            self.individuals_buffer.append(&mut tmp);
        }
    }

    fn dump_individuals_delta(
        &mut self,
        individuals: &[Individual],
        last_spend: &[f64],
        last_delta: &[i32],
        t: Option<i32>,
    ) {
        let width = self.cfg.grid_width.max(1);
        let mut seen: HashMap<i32, (usize, f64, i32, f64, i32)> = HashMap::new();
        for (i, ind) in individuals.iter().enumerate() {
            let spend = last_spend.get(i).copied().unwrap_or(0.0);
            let delta = last_delta.get(i).copied().unwrap_or(0);
            seen.insert(
                ind.id,
                (ind.position % width, ind.resources, ind.age, spend, delta),
            );
            if ensure_nn_index_entry(&mut self.nn_index_cache, ind) {
                self.nn_index_dirty = true;
            }
        }

        let need_full = self.prev_snapshot.is_empty()
            || (self.cfg.delta_full_interval > 0
                && (self.frame_count % self.cfg.delta_full_interval) == 0);

        if need_full {
            let mut writer = BufferWriter::new(&mut self.delta_buffer);
            let _ = write!(writer, "{{");
            if let Some(tv) = t {
                let _ = write!(writer, "\"t\":{}, ", tv);
            }
            let _ = write!(writer, "\"full\":true, \"individuals\":[");
            let mut first = true;
            for (id, (pos, res, age, sp, de)) in seen.iter() {
                if !first {
                    let _ = write!(writer, ", ");
                }
                first = false;
                let _ = write!(
                    writer,
                    "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"age\":{},\"last_spend\":{:.6},\"last_delta\":{}}}",
                    id,
                    pos,
                    res,
                    age,
                    sp,
                    de
                );
            }
            let _ = writeln!(writer, "]}}");
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
                        id,
                        pos,
                        res,
                        age,
                        sp,
                        de
                    ));
                }
                Some((p_pos, p_res, p_age, p_sp, p_de)) => {
                    if *p_pos != *pos
                        || (*p_de != *de)
                        || (*p_sp != *sp)
                        || (*p_res != *res)
                        || (*p_age != *age)
                    {
                        updates.push(format!(
                            "{{\"id\":{},\"pos\":{},\"resources\":{:.6},\"age\":{},\"last_spend\":{:.6},\"last_delta\":{}}}",
                            id,
                            pos,
                            res,
                            age,
                            sp,
                            de
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

        let mut writer = BufferWriter::new(&mut self.delta_buffer);
        let _ = write!(writer, "{{");
        if let Some(tv) = t {
            let _ = write!(writer, "\"t\":{}, ", tv);
        }
        let _ = write!(writer, "\"updates\":[");
        for (i, u) in updates.iter().enumerate() {
            if i > 0 {
                let _ = write!(writer, ",");
            }
            let _ = write!(writer, "{}", u);
        }
        let _ = write!(writer, "], \"remove\":[");
        for (i, rid) in removals.iter().enumerate() {
            if i > 0 {
                let _ = write!(writer, ",");
            }
            let _ = write!(writer, "{}", rid);
        }
        let _ = writeln!(writer, "]}}");

        self.prev_snapshot = seen;
        self.frame_count += 1;
    }

    pub(crate) fn flush_output_buffers(&mut self, step: i32) {
        let summary = BufferSummary {
            frames_bytes: self.frames_buffer.len(),
            env_bytes: self.env_buffer.len(),
            individuals_bytes: self.individuals_buffer.len(),
            delta_bytes: self.delta_buffer.len(),
        };
        if summary.total() == 0 {
            return;
        }
        self.log_buffer_usage(step, &summary);

        if summary.frames_bytes > 0 {
            self.open_frames_if_needed();
            if let Some(f) = self.frames_file.as_mut() {
                let _ = f.write_all(&self.frames_buffer);
                let _ = f.flush();
            }
            self.frames_buffer.clear();
        }
        if summary.env_bytes > 0 {
            self.open_env_if_needed();
            if let Some(f) = self.env_file.as_mut() {
                let _ = f.write_all(&self.env_buffer);
                let _ = f.flush();
            }
            self.env_buffer.clear();
        }
        if summary.individuals_bytes > 0 {
            self.open_individuals_if_needed();
            if let Some(f) = self.ind_file.as_mut() {
                let _ = f.write_all(&self.individuals_buffer);
                let _ = f.flush();
            }
            self.individuals_buffer.clear();
        }
        if summary.delta_bytes > 0 {
            self.open_delta_if_needed();
            if let Some(f) = self.delta_file.as_mut() {
                let _ = f.write_all(&self.delta_buffer);
                let _ = f.flush();
            }
            self.delta_buffer.clear();
        }
        self.total_bytes_flushed = self
            .total_bytes_flushed
            .saturating_add(summary.total() as u64);
        self.max_buffer_bytes = self.max_buffer_bytes.max(summary.total());
    }

    fn log_buffer_usage(&self, step: i32, summary: &BufferSummary) {
        if !self.cfg.debug_buffered_output_logs {
            return;
        }
        let total = summary.total();
        if total == 0 {
            return;
        }
        let label = if step >= 0 {
            step.to_string()
        } else {
            "final".to_string()
        };
        let detail = format!(
            "frames={} env={} indiv={} delta={}",
            summary.frames_bytes, summary.env_bytes, summary.individuals_bytes, summary.delta_bytes
        );
        let total_mib = total as f64 / (1024.0 * 1024.0);
        if let Some(budget_bytes) = self.output_budget_bytes {
            let projected = self.total_bytes_flushed + total as u64;
            let projected_mib = projected as f64 / (1024.0 * 1024.0);
            let budget_mib = budget_bytes as f64 / (1024.0 * 1024.0);
            let pct = (projected as f64 / budget_bytes as f64) * 100.0;
            if self.output_budget_warn_bytes > 0 && projected >= self.output_budget_warn_bytes {
                println!(
                    "[warn] t={} buffered {:.2} MiB (projected {:.2} MiB, {:.1}% of {:.2} MiB budget); {};",
                    label, total_mib, projected_mib, pct, budget_mib, detail
                );
            } else {
                println!(
                    "[info] t={} buffered {:.2} MiB (projected {:.2} MiB, {:.1}% of {:.2} MiB budget); {};",
                    label, total_mib, projected_mib, pct, budget_mib, detail
                );
            }
        } else {
            println!(
                "[info] t={} buffered {:.2} MiB ({} bytes); {};",
                label, total_mib, total, detail
            );
        }
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
        let do_flush = (self.cfg.nn_index_flush_interval > 0
            && t % self.cfg.nn_index_flush_interval == 0)
            || (self.cfg.nn_index_flush_size > 0
                && self.nn_index_cache.len() >= self.cfg.nn_index_flush_size);
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
        self.flush_output_buffers(-1);
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
