use super::goo_graph::GooGraphBrain;
use super::mlp::MlpBrain;
use super::perceptron::PerceptronBrain;
use super::spatial_blob::{SpatialBlobBrain, SpatialBlobLayout};
use super::spatial_graph::SpatialGraphBrain;
use crate::config::{ArchitecturePolicy, Config};
use serde::{Deserialize, Serialize};

pub const SENSOR_DIM: usize = 8;
/// Perceptron uses current-cell sensors only (resource, competition, internal stores).
pub const PERCEPTRON_SENSOR_DIM: usize = 3;
pub const THOUGHT_DIM: usize = 8;
pub const EMBED_DIM: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrainKind {
    Perceptron,
    Mlp,
    SpatialGraph,
    SpatialBlob,
    GooGraph,
}

impl BrainKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BrainKind::Perceptron => "perceptron",
            BrainKind::Mlp => "mlp",
            BrainKind::SpatialGraph => "spatial_graph",
            BrainKind::SpatialBlob => "spatial_blob",
            BrainKind::GooGraph => "goo_graph",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainInner {
    Perceptron(PerceptronBrain),
    Mlp(MlpBrain),
    SpatialGraph(SpatialGraphBrain),
    SpatialBlob(SpatialBlobBrain),
    GooGraph(GooGraphBrain),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainGenome {
    pub kind: BrainKind,
    pub inner: BrainInner,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpatialRuntime {
    pub node_embeddings: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct ForwardContext<'a> {
    pub sensors: [f64; SENSOR_DIM],
    pub position: usize,
    pub hidden_buf: &'a mut [f64],
    pub out2_buf: &'a mut [f64],
    pub spatial_runtime: Option<&'a mut SpatialRuntime>,
}

#[derive(Debug, Clone, Copy)]
pub struct BrainOutput {
    pub move_signal: f64,
    pub direction_signal: f64,
}

impl BrainGenome {
    pub fn new_founder(cfg: &Config) -> Self {
        let inputs = match cfg.architecture_policy {
            ArchitecturePolicy::PerceptronOnly => PERCEPTRON_SENSOR_DIM,
            _ => cfg.nn_inputs.max(SENSOR_DIM as i32) as usize,
        };
        let nodes = cfg.nn_nodes.max(1) as usize;

        match cfg.architecture_policy {
            ArchitecturePolicy::PerceptronOnly => BrainGenome {
                kind: BrainKind::Perceptron,
                inner: BrainInner::Perceptron(PerceptronBrain::new(inputs, nodes)),
            },
            ArchitecturePolicy::EvolveMlp => {
                let layers = cfg.nn_layers.max(1);
                let nodes2 = if layers >= 2 {
                    cfg.nn_nodes2.max(1) as usize
                } else {
                    0
                };
                BrainGenome {
                    kind: BrainKind::Mlp,
                    inner: BrainInner::Mlp(MlpBrain::from_config(
                        nodes, inputs, layers, nodes2,
                    )),
                }
            }
            ArchitecturePolicy::EvolveSpatialGraph => {
                if cfg.uses_legacy_spatial_graph() {
                    BrainGenome {
                        kind: BrainKind::SpatialGraph,
                        inner: BrainInner::SpatialGraph(SpatialGraphBrain::new(nodes)),
                    }
                } else {
                    let layout = SpatialBlobLayout::from_config_str(&cfg.spatial_blob_layout);
                    let layers = if cfg.nn_layers > 1 {
                        cfg.nn_layers
                    } else {
                        cfg.nn_nodes.max(1)
                    } as usize;
                    BrainGenome {
                        kind: BrainKind::SpatialBlob,
                        inner: BrainInner::SpatialBlob(SpatialBlobBrain::new(layout, layers)),
                    }
                }
            }
            ArchitecturePolicy::EvolveGoo => BrainGenome {
                kind: BrainKind::GooGraph,
                inner: BrainInner::GooGraph(GooGraphBrain::new_initial()),
            },
        }
    }

    pub fn kind(&self) -> BrainKind {
        self.kind
    }

    pub fn complexity(&self) -> usize {
        match &self.inner {
            BrainInner::Perceptron(b) => b.complexity(),
            BrainInner::Mlp(b) => b.complexity(),
            BrainInner::SpatialGraph(b) => b.complexity(),
            BrainInner::SpatialBlob(b) => b.complexity(),
            BrainInner::GooGraph(b) => b.complexity(),
        }
    }

    pub fn max_buffer_nodes(&self) -> usize {
        match &self.inner {
            BrainInner::Perceptron(b) => b.nodes,
            BrainInner::Mlp(b) => b.nn.nodes.max(b.nn.nodes2),
            BrainInner::SpatialGraph(b) => b.landmark_count().max(4),
            BrainInner::SpatialBlob(b) => b.nodes_per_layer().max(4),
            BrainInner::GooGraph(b) => b.node_count().max(THOUGHT_DIM),
        }
    }

    pub fn forward(&self, ctx: ForwardContext<'_>) -> BrainOutput {
        match &self.inner {
            BrainInner::Perceptron(b) => b.forward(&ctx.sensors, ctx.hidden_buf),
            BrainInner::Mlp(b) => b.forward(&ctx.sensors, ctx.hidden_buf, ctx.out2_buf),
            BrainInner::SpatialGraph(b) => {
                let runtime = ctx
                    .spatial_runtime
                    .expect("spatial runtime required for SpatialGraph brain");
                b.forward(&ctx.sensors, ctx.position, runtime, ctx.hidden_buf)
            }
            BrainInner::SpatialBlob(b) => b.forward(&ctx.sensors, ctx.hidden_buf),
            BrainInner::GooGraph(b) => b.forward(&ctx.sensors, ctx.hidden_buf),
        }
    }

    pub fn as_mlp(&self) -> Option<&MlpBrain> {
        match &self.inner {
            BrainInner::Mlp(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_mlp_mut(&mut self) -> Option<&mut MlpBrain> {
        match &mut self.inner {
            BrainInner::Mlp(b) => Some(b),
            _ => None,
        }
    }

    pub fn legacy_nn_view(&self) -> Option<&crate::neural_network::NeuralNetwork> {
        self.as_mlp().map(|m| &m.nn)
    }

    pub fn init_spatial_runtime(&self) -> Option<SpatialRuntime> {
        match &self.inner {
            BrainInner::SpatialGraph(b) => Some(b.init_runtime()),
            _ => None,
        }
    }

    pub fn genetic_distance(&self, other: &BrainGenome) -> f64 {
        if self.kind != other.kind {
            return 1.0 + param_distance(self, other);
        }
        match (&self.inner, &other.inner) {
            (BrainInner::Perceptron(a), BrainInner::Perceptron(b)) => a.param_distance(b),
            (BrainInner::Mlp(a), BrainInner::Mlp(b)) => a.param_distance(b),
            (BrainInner::SpatialGraph(a), BrainInner::SpatialGraph(b)) => a.distance(b),
            (BrainInner::SpatialBlob(a), BrainInner::SpatialBlob(b)) => a.distance(b),
            (BrainInner::GooGraph(a), BrainInner::GooGraph(b)) => a.distance(b),
            _ => 1.0,
        }
    }

    pub fn structure_signature(&self) -> String {
        match &self.inner {
            BrainInner::Perceptron(b) => format!("p:{}:{}", b.inputs, b.nodes),
            BrainInner::Mlp(b) => {
                let (n, i, n2, l) = b.nn.shape();
                format!("m:{}:{}:{}:{}", n, i, n2, l)
            }
            BrainInner::SpatialGraph(b) => b.structure_signature(),
            BrainInner::SpatialBlob(b) => b.structure_signature(),
            BrainInner::GooGraph(b) => b.structure_signature(),
        }
    }

    /// Lightweight structure signature from serialized brain JSON (no weight tensors).
    pub fn structure_signature_from_brain_json(json: &str) -> Option<String> {
        let v = serde_json::from_str::<serde_json::Value>(json).ok()?;
        let kind = v.get("kind")?.as_str()?;
        let inner = v.get("inner")?;
        match kind {
            "perceptron" => {
                let p = inner.get("Perceptron")?;
                Some(format!(
                    "p:{}:{}",
                    p.get("inputs")?.as_u64()?,
                    p.get("nodes")?.as_u64()?
                ))
            }
            "mlp" => {
                let m = inner.get("Mlp")?;
                Some(format!(
                    "m:{}:{}:{}:{}",
                    m.get("nodes")?.as_u64()?,
                    m.get("inputs")?.as_u64()?,
                    m.get("nodes2")?.as_u64()?,
                    m.get("layers")?.as_i64()?
                ))
            }
            "spatial_graph" => {
                let s = inner.get("SpatialGraph")?;
                let landmark_count = s.get("landmark_count")?.as_u64()? as usize;
                let edges = s.get("edges").and_then(|e| e.as_array()).map(|a| a.len()).unwrap_or(0);
                let message_passes = s.get("message_passes")?.as_u64()?;
                let masks: Vec<String> = s
                    .get("landmark_sectors")
                    .and_then(|m| m.as_array())
                    .map(|arr| {
                        arr.iter()
                            .take(landmark_count)
                            .filter_map(|v| v.as_u64().map(|n| format!("{n:02x}")))
                            .collect()
                    })
                    .unwrap_or_default();
                Some(format!(
                    "sg:{}:{}:{}:[{}]",
                    landmark_count,
                    edges,
                    message_passes,
                    masks.join(",")
                ))
            }
            "spatial_blob" => {
                let b = inner.get("SpatialBlob")?;
                let layout = b.get("layout")?.as_str()?;
                let layer_count = b.get("layer_count")?.as_u64()?;
                let nodes = if layout.contains("cardinal") { 4 } else { 6 };
                Some(format!("blob:{layout}:{layer_count}:{nodes}"))
            }
            "goo_graph" => {
                let g = inner.get("GooGraph")?;
                let nodes = g.get("nodes")?.as_array()?;
                let ops: Vec<String> = nodes
                    .iter()
                    .filter_map(|n| n.get("op").and_then(|o| o.as_str()))
                    .map(goo_op_debug_label)
                    .collect();
                Some(format!("goo:{}:[{}]", nodes.len(), ops.join(",")))
            }
            _ => None,
        }
    }
}

fn goo_op_debug_label(snake: &str) -> String {
    match snake {
        "generate" => "Generate".to_string(),
        "aggregate" => "Aggregate".to_string(),
        "score" => "Score".to_string(),
        "keep_best" => "KeepBest".to_string(),
        other => other.to_string(),
    }
}

fn param_distance(a: &BrainGenome, b: &BrainGenome) -> f64 {
    let pa = a.flatten_params();
    let pb = b.flatten_params();
    let n = pa.len().max(pb.len()).max(1);
    let mut sum = 0.0;
    for i in 0..n {
        let da = pa.get(i).copied().unwrap_or(0.0);
        let db = pb.get(i).copied().unwrap_or(0.0);
        sum += (da - db).powi(2);
    }
    (sum / n as f64).sqrt()
}

impl BrainGenome {
    pub fn flatten_params(&self) -> Vec<f64> {
        match &self.inner {
            BrainInner::Perceptron(b) => b.flatten_params(),
            BrainInner::Mlp(b) => b.flatten_params(),
            BrainInner::SpatialGraph(b) => b.flatten_params(),
            BrainInner::SpatialBlob(b) => b.flatten_params(),
            BrainInner::GooGraph(b) => b.flatten_params(),
        }
    }
}
