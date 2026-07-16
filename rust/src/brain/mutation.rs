use super::genome::{BrainGenome, BrainInner, BrainKind};
use super::goo_graph::GooGraphBrain;
use super::mlp::MlpBrain;
use super::perceptron::PerceptronBrain;
use super::spatial_blob::{SpatialBlobBrain, SpatialBlobLayout};
use super::spatial_graph::SpatialGraphBrain;
use crate::config::{ArchitecturePolicy, Config};
use rand::Rng;
use rand_distr::{Cauchy, Distribution, Uniform};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralMutation {
    None,
    ParametricOnly,
    KindChange { from: BrainKind, to: BrainKind },
    MlpAddNodeLayer1,
    MlpAddNodeLayer2,
    MlpAddLayer,
    MlpRemoveNode,
    SpatialAddLandmark,
    SpatialAddEdge,
    SpatialRemoveEdge,
    SpatialMutateSector,
    SpatialBlobAddLayer,
    GooAddGenerate,
    GooAddAggregate,
    GooAddScore,
    GooAddKeepBest,
    GooChangeOp,
}

pub fn mutate_parametric(brain: &mut BrainGenome, rng: &mut impl Rng, rate: f64, sigma: f64) {
    let cauchy = Cauchy::new(0.0, sigma).unwrap();
    match &mut brain.inner {
        BrainInner::Perceptron(b) => {
            for w in b.weights.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.biases.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
        }
        BrainInner::Mlp(b) => {
            let nn = &mut b.nn;
            for w in nn.weights.iter_mut().chain(nn.weights_multi.iter_mut()) {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in nn.biases.iter_mut().chain(nn.biases2.iter_mut()) {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in nn.weights2.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
        }
        BrainInner::SpatialGraph(b) => {
            for w in b.edge_weights.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.readout_weights.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.readout_biases.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
        }
        BrainInner::SpatialBlob(b) => {
            for w in b.layer0_input_weights.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.layer_recurrent_weights.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.layer_biases.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.readout_weights.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in b.readout_biases.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
        }
        BrainInner::GooGraph(b) => {
            for w in b.params.iter_mut() {
                if rng.gen::<f64>() < rate {
                    *w += cauchy.sample(rng);
                }
            }
        }
    }
}

pub fn reproduce_brain(
    parent: &BrainGenome,
    cfg: &Config,
    rng: &mut impl Rng,
) -> (BrainGenome, StructuralMutation) {
    let mut child = parent.clone();
    mutate_parametric(
        &mut child,
        rng,
        cfg.mutation_rate,
        cfg.mutation_sigma,
    );

    if rng.gen::<f64>() >= cfg.architecture_mutation_rate {
        return (child, StructuralMutation::ParametricOnly);
    }

    let max_nodes = cfg.max_brain_nodes.max(1) as usize;
    if child.complexity() >= max_nodes {
        return (child, StructuralMutation::ParametricOnly);
    }

    match cfg.architecture_policy {
        ArchitecturePolicy::PerceptronOnly => {
            (child, StructuralMutation::None)
        }
        ArchitecturePolicy::EvolveMlp => {
            apply_mlp_structural(&mut child, parent, cfg, rng)
        }
        ArchitecturePolicy::EvolveSpatialGraph => {
            apply_spatial_structural(&mut child, parent, cfg, rng)
        }
        ArchitecturePolicy::EvolveGoo => {
            apply_goo_structural(&mut child, parent, cfg, rng)
        }
    }
}

fn perceptron_to_mlp(p: &PerceptronBrain) -> MlpBrain {
    MlpBrain::from_perceptron(p.inputs, p.nodes, p.weights.clone(), p.biases.clone())
}

fn apply_mlp_structural(
    child: &mut BrainGenome,
    parent: &BrainGenome,
    cfg: &Config,
    rng: &mut impl Rng,
) -> (BrainGenome, StructuralMutation) {
    if child.kind == BrainKind::Perceptron {
        if let BrainInner::Perceptron(ref p) = child.inner {
            let mlp = perceptron_to_mlp(p);
            child.kind = BrainKind::Mlp;
            child.inner = BrainInner::Mlp(mlp);
            return (
                child.clone(),
                StructuralMutation::KindChange {
                    from: BrainKind::Perceptron,
                    to: BrainKind::Mlp,
                },
            );
        }
    }

    if let BrainInner::Mlp(ref mut mlp) = child.inner {
        let dup = cfg.structural_duplication;
        let sigma = cfg.mutation_sigma;
        let choice = rng.gen_range(0..4);
        match choice {
            0 => {
                if dup {
                    mlp.duplicate_hidden_node_layer1(rng, sigma);
                } else {
                    mlp.add_hidden_node_layer1(rng);
                }
                return (child.clone(), StructuralMutation::MlpAddNodeLayer1);
            }
            1 if mlp.nn.layers == 2 => {
                if dup {
                    mlp.duplicate_hidden_node_layer2(rng, sigma);
                } else {
                    mlp.add_hidden_node_layer2(rng);
                }
                return (child.clone(), StructuralMutation::MlpAddNodeLayer2);
            }
            2 if mlp.nn.layers == 1 => {
                let n2 = cfg.nn_nodes2.max(1) as usize;
                if dup {
                    mlp.duplicate_second_layer(rng, n2, sigma);
                } else {
                    mlp.add_second_layer(rng, n2);
                }
                return (child.clone(), StructuralMutation::MlpAddLayer);
            }
            _ => {
                if mlp.remove_hidden_node_layer1() {
                    return (child.clone(), StructuralMutation::MlpRemoveNode);
                }
            }
        }
    }

    let _ = parent;
    (child.clone(), StructuralMutation::ParametricOnly)
}

fn apply_spatial_structural(
    child: &mut BrainGenome,
    _parent: &BrainGenome,
    cfg: &Config,
    rng: &mut impl Rng,
) -> (BrainGenome, StructuralMutation) {
    if child.kind == BrainKind::Perceptron {
        if let BrainInner::Perceptron(ref p) = child.inner {
            if cfg.uses_legacy_spatial_graph() {
                let mut sg = SpatialGraphBrain::new(1);
                sg.readout_weights = p.weights.clone();
                sg.readout_biases = p.biases.clone();
                child.kind = BrainKind::SpatialGraph;
                child.inner = BrainInner::SpatialGraph(sg);
                return (
                    child.clone(),
                    StructuralMutation::KindChange {
                        from: BrainKind::Perceptron,
                        to: BrainKind::SpatialGraph,
                    },
                );
            }
            let layout = SpatialBlobLayout::from_config_str(&cfg.spatial_blob_layout);
            let mut blob = SpatialBlobBrain::new(layout, 1);
            blob.init_founder_weights(rng);
            child.kind = BrainKind::SpatialBlob;
            child.inner = BrainInner::SpatialBlob(blob);
            return (
                child.clone(),
                StructuralMutation::KindChange {
                    from: BrainKind::Perceptron,
                    to: BrainKind::SpatialBlob,
                },
            );
        }
    }

    if let BrainInner::SpatialBlob(ref mut blob) = child.inner {
        let max_layers = cfg.max_brain_nodes.max(1) as usize;
        if blob.layer_count < max_layers {
            blob.add_layer(rng, cfg.structural_duplication, cfg.mutation_sigma);
            return (child.clone(), StructuralMutation::SpatialBlobAddLayer);
        }
    }

    if let BrainInner::SpatialGraph(ref mut sg) = child.inner {
        let dup = cfg.structural_duplication;
        let sigma = cfg.mutation_sigma;
        match rng.gen_range(0..4) {
            0 => {
                if dup {
                    sg.duplicate_landmark(rng, sigma);
                } else {
                    sg.add_landmark(rng);
                }
                return (child.clone(), StructuralMutation::SpatialAddLandmark);
            }
            1 => {
                if dup {
                    sg.duplicate_edge(rng, sigma);
                } else {
                    sg.add_edge(rng);
                }
                return (child.clone(), StructuralMutation::SpatialAddEdge);
            }
            2 => {
                if sg.mutate_sector_mask(rng) {
                    return (child.clone(), StructuralMutation::SpatialMutateSector);
                }
            }
            _ => {
                if sg.remove_edge() {
                    return (child.clone(), StructuralMutation::SpatialRemoveEdge);
                }
            }
        }
    }

    (child.clone(), StructuralMutation::ParametricOnly)
}

fn apply_goo_structural(
    child: &mut BrainGenome,
    _parent: &BrainGenome,
    cfg: &Config,
    rng: &mut impl Rng,
) -> (BrainGenome, StructuralMutation) {
    if child.kind == BrainKind::Perceptron {
        if let BrainInner::Perceptron(ref p) = child.inner {
            let mut goo = GooGraphBrain::new_initial();
            goo.params = p.weights.clone();
            child.kind = BrainKind::GooGraph;
            child.inner = BrainInner::GooGraph(goo);
            return (
                child.clone(),
                StructuralMutation::KindChange {
                    from: BrainKind::Perceptron,
                    to: BrainKind::GooGraph,
                },
            );
        }
    }

    if let BrainInner::GooGraph(ref mut goo) = child.inner {
        let dup = cfg.structural_duplication;
        let sigma = cfg.mutation_sigma;
        let n = goo.nodes.len();
        match rng.gen_range(0..5) {
            0 => {
                let parent_id = if n > 0 {
                    Some(rng.gen_range(0..n))
                } else {
                    None
                };
                if dup {
                    goo.duplicate_generate_node(rng, parent_id, sigma);
                } else {
                    goo.add_generate_node(rng, parent_id);
                }
                return (child.clone(), StructuralMutation::GooAddGenerate);
            }
            1 if n >= 2 => {
                let parents = GooGraphBrain::sample_parent_ids(rng, n, 2);
                if parents.len() >= 2 {
                    goo.add_aggregate_node(rng, parents);
                    return (child.clone(), StructuralMutation::GooAddAggregate);
                }
            }
            2 if n >= 1 => {
                goo.add_score_node(rng.gen_range(0..n));
                return (child.clone(), StructuralMutation::GooAddScore);
            }
            3 if n >= 2 => {
                let parent_count = rng.gen_range(2..=n.min(4));
                let parents = GooGraphBrain::sample_parent_ids(rng, n, parent_count);
                if parents.len() >= 2 {
                    goo.add_keep_best_node(parents);
                    return (child.clone(), StructuralMutation::GooAddKeepBest);
                }
            }
            _ => {
                if n > 0 {
                    goo.change_op_type(rng.gen_range(0..n), rng);
                    return (child.clone(), StructuralMutation::GooChangeOp);
                }
            }
        }
    }

    (child.clone(), StructuralMutation::ParametricOnly)
}

/// Targeted structural / parametric ops for the lab playground (not random reproduction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabStructuralOp {
    Parametric,
    MlpAddNodeLayer1,
    MlpAddNodeLayer2,
    MlpAddLayer,
    MlpRemoveNode,
    SpatialAddLandmark,
    SpatialAddEdge,
    SpatialRemoveEdge,
    GooAddGenerate,
    GooAddAggregate,
    GooAddScore,
    GooAddKeepBest,
    GooChangeOp,
}

impl LabStructuralOp {
    pub const ALL: &'static [LabStructuralOp] = &[
        LabStructuralOp::Parametric,
        LabStructuralOp::MlpAddNodeLayer1,
        LabStructuralOp::MlpAddNodeLayer2,
        LabStructuralOp::MlpAddLayer,
        LabStructuralOp::MlpRemoveNode,
        LabStructuralOp::SpatialAddLandmark,
        LabStructuralOp::SpatialAddEdge,
        LabStructuralOp::SpatialRemoveEdge,
        LabStructuralOp::GooAddGenerate,
        LabStructuralOp::GooAddAggregate,
        LabStructuralOp::GooAddScore,
        LabStructuralOp::GooAddKeepBest,
        LabStructuralOp::GooChangeOp,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LabStructuralOp::Parametric => "Parametric",
            LabStructuralOp::MlpAddNodeLayer1 => "MLP +node L1",
            LabStructuralOp::MlpAddNodeLayer2 => "MLP +node L2",
            LabStructuralOp::MlpAddLayer => "MLP +layer",
            LabStructuralOp::MlpRemoveNode => "MLP −node",
            LabStructuralOp::SpatialAddLandmark => "Spatial +landmark",
            LabStructuralOp::SpatialAddEdge => "Spatial +edge",
            LabStructuralOp::SpatialRemoveEdge => "Spatial −edge",
            LabStructuralOp::GooAddGenerate => "GoO +generate",
            LabStructuralOp::GooAddAggregate => "GoO +aggregate",
            LabStructuralOp::GooAddScore => "GoO +score",
            LabStructuralOp::GooAddKeepBest => "GoO +keep_best",
            LabStructuralOp::GooChangeOp => "GoO change op",
        }
    }
}

/// Apply one lab mutation to `brain`. Returns the resulting structural tag (or ParametricOnly).
pub fn apply_lab_structural(
    brain: &mut BrainGenome,
    op: LabStructuralOp,
    cfg: &Config,
    rng: &mut impl Rng,
) -> StructuralMutation {
    match op {
        LabStructuralOp::Parametric => {
            mutate_parametric(brain, rng, cfg.mutation_rate, cfg.mutation_sigma);
            StructuralMutation::ParametricOnly
        }
        LabStructuralOp::MlpAddNodeLayer1 => {
            if brain.kind == BrainKind::Perceptron {
                if let BrainInner::Perceptron(ref p) = brain.inner {
                    let mlp = perceptron_to_mlp(p);
                    brain.kind = BrainKind::Mlp;
                    brain.inner = BrainInner::Mlp(mlp);
                }
            }
            if let BrainInner::Mlp(ref mut mlp) = brain.inner {
                mlp.add_hidden_node_layer1(rng);
                return StructuralMutation::MlpAddNodeLayer1;
            }
            StructuralMutation::None
        }
        LabStructuralOp::MlpAddNodeLayer2 => {
            if let BrainInner::Mlp(ref mut mlp) = brain.inner {
                if mlp.nn.layers == 2 {
                    mlp.add_hidden_node_layer2(rng);
                    return StructuralMutation::MlpAddNodeLayer2;
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::MlpAddLayer => {
            if let BrainInner::Mlp(ref mut mlp) = brain.inner {
                if mlp.nn.layers == 1 {
                    let n2 = cfg.nn_nodes2.max(1) as usize;
                    mlp.add_second_layer(rng, n2);
                    return StructuralMutation::MlpAddLayer;
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::MlpRemoveNode => {
            if let BrainInner::Mlp(ref mut mlp) = brain.inner {
                if mlp.remove_hidden_node_layer1() {
                    return StructuralMutation::MlpRemoveNode;
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::SpatialAddLandmark => {
            if brain.kind == BrainKind::Perceptron {
                if let BrainInner::Perceptron(ref p) = brain.inner {
                    let mut sg = SpatialGraphBrain::new(1);
                    sg.readout_weights = p.weights.clone();
                    sg.readout_biases = p.biases.clone();
                    brain.kind = BrainKind::SpatialGraph;
                    brain.inner = BrainInner::SpatialGraph(sg);
                }
            }
            if let BrainInner::SpatialGraph(ref mut sg) = brain.inner {
                sg.add_landmark(rng);
                return StructuralMutation::SpatialAddLandmark;
            }
            StructuralMutation::None
        }
        LabStructuralOp::SpatialAddEdge => {
            if let BrainInner::SpatialGraph(ref mut sg) = brain.inner {
                sg.add_edge(rng);
                return StructuralMutation::SpatialAddEdge;
            }
            StructuralMutation::None
        }
        LabStructuralOp::SpatialRemoveEdge => {
            if let BrainInner::SpatialGraph(ref mut sg) = brain.inner {
                if sg.remove_edge() {
                    return StructuralMutation::SpatialRemoveEdge;
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::GooAddGenerate => {
            if brain.kind == BrainKind::Perceptron {
                if let BrainInner::Perceptron(ref p) = brain.inner {
                    let mut goo = GooGraphBrain::new_initial();
                    goo.params = p.weights.clone();
                    brain.kind = BrainKind::GooGraph;
                    brain.inner = BrainInner::GooGraph(goo);
                }
            }
            if let BrainInner::GooGraph(ref mut goo) = brain.inner {
                let n = goo.nodes.len();
                let parent_id = if n > 0 {
                    Some(rng.gen_range(0..n))
                } else {
                    None
                };
                goo.add_generate_node(rng, parent_id);
                return StructuralMutation::GooAddGenerate;
            }
            StructuralMutation::None
        }
        LabStructuralOp::GooAddAggregate => {
            if let BrainInner::GooGraph(ref mut goo) = brain.inner {
                let n = goo.nodes.len();
                if n >= 2 {
                    let parents = GooGraphBrain::sample_parent_ids(rng, n, 2);
                    if parents.len() >= 2 {
                        goo.add_aggregate_node(rng, parents);
                        return StructuralMutation::GooAddAggregate;
                    }
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::GooAddScore => {
            if let BrainInner::GooGraph(ref mut goo) = brain.inner {
                let n = goo.nodes.len();
                if n >= 1 {
                    goo.add_score_node(rng.gen_range(0..n));
                    return StructuralMutation::GooAddScore;
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::GooAddKeepBest => {
            if let BrainInner::GooGraph(ref mut goo) = brain.inner {
                let n = goo.nodes.len();
                if n >= 2 {
                    let parent_count = rng.gen_range(2..=n.min(4));
                    let parents = GooGraphBrain::sample_parent_ids(rng, n, parent_count);
                    if parents.len() >= 2 {
                        goo.add_keep_best_node(parents);
                        return StructuralMutation::GooAddKeepBest;
                    }
                }
            }
            StructuralMutation::None
        }
        LabStructuralOp::GooChangeOp => {
            if let BrainInner::GooGraph(ref mut goo) = brain.inner {
                let n = goo.nodes.len();
                if n > 0 {
                    goo.change_op_type(rng.gen_range(0..n), rng);
                    return StructuralMutation::GooChangeOp;
                }
            }
            StructuralMutation::None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FounderWeightMode {
    Neutral,
    RandomUniform,
    RandomSmall,
}

impl FounderWeightMode {
    pub fn label(self) -> &'static str {
        match self {
            FounderWeightMode::Neutral => "neutral",
            FounderWeightMode::RandomUniform => "random_uniform",
            FounderWeightMode::RandomSmall => "random_small",
        }
    }

    pub fn parse_list(raw: &str) -> Vec<Self> {
        let mut out = Vec::new();
        for part in raw.split(',') {
            let p = part.trim().to_ascii_lowercase();
            let mode = match p.as_str() {
                "neutral" | "zero" => Some(FounderWeightMode::Neutral),
                "random" | "random_uniform" | "uniform" => Some(FounderWeightMode::RandomUniform),
                "random_small" | "small" => Some(FounderWeightMode::RandomSmall),
                _ => None,
            };
            if let Some(m) = mode {
                if !out.contains(&m) {
                    out.push(m);
                }
            }
        }
        if out.is_empty() {
            out.push(FounderWeightMode::Neutral);
        }
        out
    }
}

pub fn init_founder_weights_with_mode(
    brain: &mut BrainGenome,
    rng: &mut impl Rng,
    mode: FounderWeightMode,
    cfg: &Config,
) {
    match mode {
        FounderWeightMode::Neutral => {
            if cfg.radiate > 0 {
                let cauchy = Cauchy::new(0.0, cfg.mutation_sigma).unwrap();
                for _ in 0..cfg.radiate.max(0) as usize {
                    mutate_parametric(brain, rng, cfg.mutation_rate, cfg.mutation_sigma);
                    let _ = cauchy;
                }
            } else if let BrainInner::SpatialBlob(ref mut b) = brain.inner {
                b.init_founder_layers(rng);
            } else {
                zero_brain_weights(brain);
            }
        }
        FounderWeightMode::RandomUniform | FounderWeightMode::RandomSmall => {
            let span = if mode == FounderWeightMode::RandomSmall {
                0.12
            } else {
                1.0
            };
            let wrand = Uniform::new(-span, span);
            match &mut brain.inner {
                BrainInner::Perceptron(b) => {
                    for w in b.weights.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                    for w in b.biases.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                }
                BrainInner::Mlp(b) => {
                    let nn = &mut b.nn;
                    for w in nn.weights.iter_mut().chain(nn.weights_multi.iter_mut()) {
                        *w = wrand.sample(rng);
                    }
                    for w in nn.biases.iter_mut().chain(nn.biases2.iter_mut()) {
                        *w = wrand.sample(rng);
                    }
                    for w in nn.weights2.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                }
                BrainInner::SpatialGraph(b) => {
                    for w in b.edge_weights.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                    for w in b.readout_weights.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                    for w in b.readout_biases.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                }
                BrainInner::SpatialBlob(b) => {
                    b.init_founder_weights(rng);
                }
                BrainInner::GooGraph(b) => {
                    for w in b.params.iter_mut() {
                        *w = wrand.sample(rng);
                    }
                }
            }
        }
    }
}

pub fn init_founder_weights(brain: &mut BrainGenome, rng: &mut impl Rng, cfg: &Config) {
    if let BrainInner::SpatialBlob(ref mut b) = brain.inner {
        if cfg.neutral_founder_weights && cfg.radiate <= 0 {
            b.init_founder_layers(rng);
        } else {
            b.init_founder_weights(rng);
        }
        if cfg.radiate > 0 {
            let cauchy = Cauchy::new(0.0, cfg.mutation_sigma).unwrap();
            for _ in 0..cfg.radiate.max(0) as usize {
                mutate_parametric(brain, rng, cfg.mutation_rate, cfg.mutation_sigma);
                let _ = cauchy;
            }
        }
        return;
    }
    if cfg.neutral_founder_weights && cfg.radiate <= 0 {
        zero_brain_weights(brain);
        return;
    }
    let wrand = Uniform::new(-1.0f64, 1.0f64);
    if cfg.radiate > 0 {
        let events = cfg.radiate.max(0) as usize;
        let cauchy = Cauchy::new(0.0, cfg.mutation_sigma).unwrap();
        for _ in 0..events {
            mutate_parametric(brain, rng, cfg.mutation_rate, cfg.mutation_sigma);
            let _ = cauchy;
        }
    } else {
        match &mut brain.inner {
            BrainInner::Perceptron(b) => {
                for w in b.weights.iter_mut() {
                    *w = wrand.sample(rng);
                }
                for w in b.biases.iter_mut() {
                    *w = wrand.sample(rng);
                }
            }
            BrainInner::Mlp(b) => {
                let nn = &mut b.nn;
                for w in nn.weights.iter_mut().chain(nn.weights_multi.iter_mut()) {
                    *w = wrand.sample(rng);
                }
                for w in nn.biases.iter_mut().chain(nn.biases2.iter_mut()) {
                    *w = wrand.sample(rng);
                }
                for w in nn.weights2.iter_mut() {
                    *w = wrand.sample(rng);
                }
            }
            BrainInner::SpatialGraph(b) => {
                for w in b.edge_weights.iter_mut() {
                    *w = wrand.sample(rng);
                }
                for w in b.readout_weights.iter_mut() {
                    *w = wrand.sample(rng);
                }
            }
            BrainInner::SpatialBlob(b) => {
                for w in b.layer0_input_weights.iter_mut() {
                    *w = wrand.sample(rng);
                }
                for w in b.layer_recurrent_weights.iter_mut() {
                    *w = wrand.sample(rng);
                }
                for w in b.layer_biases.iter_mut() {
                    *w = wrand.sample(rng);
                }
                for w in b.readout_weights.iter_mut() {
                    *w = wrand.sample(rng);
                }
                for w in b.readout_biases.iter_mut() {
                    *w = wrand.sample(rng);
                }
            }
            BrainInner::GooGraph(b) => {
                for w in b.params.iter_mut() {
                    *w = wrand.sample(rng);
                }
            }
        }
    }
}

/// Zero all founder parameters (spatial / GoO struct defaults are non-zero until mutated).
fn zero_brain_weights(brain: &mut BrainGenome) {
    match &mut brain.inner {
        BrainInner::Perceptron(b) => {
            b.weights.fill(0.0);
            b.biases.fill(0.0);
        }
        BrainInner::Mlp(b) => {
            let nn = &mut b.nn;
            nn.weights.fill(0.0);
            nn.weights_multi.fill(0.0);
            nn.biases.fill(0.0);
            nn.weights2.fill(0.0);
            nn.biases2.fill(0.0);
        }
        BrainInner::SpatialGraph(b) => {
            b.edge_weights.fill(0.0);
            b.readout_weights.fill(0.0);
            b.readout_biases.fill(0.0);
        }
        BrainInner::SpatialBlob(b) => {
            b.layer0_input_weights.fill(0.0);
            b.layer_recurrent_weights.fill(0.0);
            b.layer_biases.fill(0.0);
            b.readout_weights.fill(0.0);
            b.readout_biases.fill(0.0);
        }
        BrainInner::GooGraph(b) => {
            b.params.fill(0.0);
        }
    }
}
