use crate::brain::{
    init_founder_weights, reproduce_brain, BrainGenome, ForwardContext, SpatialRuntime,
    StructuralMutation,
};
use crate::config::Config;
use crate::neural_network::NeuralNetwork;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct ActionDecision {
    pub move_one_step: bool,
    pub move_signal: f64,
    pub direction_degrees: f64,
    pub movement_spend: f64,
    pub gathered: f64,
}

#[derive(Debug, Clone)]
pub struct ReproductionResult {
    pub child: Individual,
    pub structural_mutation: StructuralMutation,
    pub genetic_distance: f64,
}

#[derive(Clone, Debug)]
pub struct Individual {
    pub brain: BrainGenome,
    pub spatial_runtime: Option<SpatialRuntime>,
    pub resources: f64,
    pub position: usize,
    pub id: i32,
    pub age: i32,
    pub lineage_color: [u8; 3],
    pub parent_id: Option<i32>,
    /// Steps remaining before the next hex move is allowed (0 = ready).
    pub move_cooldown: u32,
}

impl Individual {
    pub const MAINTENANCE_COST_PER_NODE: f64 = 0.01;
    pub const REPRODUCTION_THRESHOLD: f64 = 20.0;
    pub const REPRODUCTION_COST: f64 = 5.0;
    pub const MUTATION_RATE: f64 = 0.01;
    pub const MUTATION_SIGMA: f64 = 0.1;
    pub const MOVEMENT_COST_FACTOR: f64 = 0.1;

    pub fn new(brain: BrainGenome, pos: usize, id: i32) -> Self {
        let spatial_runtime = brain.init_spatial_runtime();
        Individual {
            brain,
            spatial_runtime,
            resources: 10.0,
            position: pos,
            id,
            age: 0,
            lineage_color: [128, 128, 128],
            parent_id: None,
            move_cooldown: 0,
        }
    }

    pub fn new_founder(cfg: &Config, pos: usize, id: i32, rng: &mut impl Rng) -> Self {
        let mut brain = BrainGenome::new_founder(cfg);
        init_founder_weights(&mut brain, rng, cfg);
        Self::new(brain, pos, id)
    }

    pub fn new_founder_with_mode(
        cfg: &Config,
        pos: usize,
        id: i32,
        rng: &mut impl Rng,
        mode: crate::brain::FounderWeightMode,
    ) -> Self {
        let mut brain = BrainGenome::new_founder(cfg);
        crate::brain::init_founder_weights_with_mode(&mut brain, rng, mode, cfg);
        Self::new(brain, pos, id)
    }

    pub fn new_with_color(brain: BrainGenome, pos: usize, id: i32, color: [u8; 3]) -> Self {
        let mut ind = Self::new(brain, pos, id);
        ind.lineage_color = color;
        ind
    }

    pub fn from_legacy_nn(nn: NeuralNetwork, pos: usize, id: i32) -> Self {
        use crate::brain::{BrainGenome, BrainInner, BrainKind, MlpBrain};
        Self::new(
            BrainGenome {
                kind: BrainKind::Mlp,
                inner: BrainInner::Mlp(MlpBrain { nn }),
            },
            pos,
            id,
        )
    }

    pub fn from_legacy_nn_with_color(
        nn: NeuralNetwork,
        pos: usize,
        id: i32,
        color: [u8; 3],
    ) -> Self {
        let mut ind = Self::from_legacy_nn(nn, pos, id);
        ind.lineage_color = color;
        ind
    }

    /// Legacy accessor for MLP-backed individuals (tensor/GPU path, webviewer).
    pub fn nn(&self) -> &NeuralNetwork {
        static EMPTY: std::sync::OnceLock<NeuralNetwork> = std::sync::OnceLock::new();
        self.brain
            .legacy_nn_view()
            .unwrap_or_else(|| EMPTY.get_or_init(|| NeuralNetwork::new(1, 8, 1, 0)))
    }

    pub fn nn_mut(&mut self) -> Option<&mut NeuralNetwork> {
        self.brain.as_mlp_mut().map(|m| &mut m.nn)
    }

    pub fn complexity(&self) -> usize {
        self.brain.complexity()
    }

    #[inline]
    fn direction_signal_to_degrees(signal: f64) -> f64 {
        let normalized = ((signal.tanh() + 1.0) * 0.5).clamp(0.0, 1.0);
        normalized * 360.0
    }

    pub fn decide_action(
        &mut self,
        edge_resources: [f64; 6],
        local_population: i32,
        movement_cost_factor: f64,
        _rng: &mut impl Rng,
        hidden_buf: &mut [f64],
        out2_buf: &mut [f64],
    ) -> ActionDecision {
        use crate::step_engine::build_sensor_inputs_for_kind;

        let center = edge_resources.iter().sum::<f64>() / 6.0;
        let sensors = build_sensor_inputs_for_kind(
            self.brain.kind(),
            edge_resources,
            center,
            local_population,
            self.resources,
        );

        let ctx = ForwardContext {
            sensors,
            position: self.position,
            hidden_buf,
            out2_buf,
            spatial_runtime: self.spatial_runtime.as_mut(),
        };
        let output = self.brain.forward(ctx);

        let direction_degrees = Self::direction_signal_to_degrees(output.direction_signal);
        if output.move_signal <= 0.0 {
            let denom = (local_population.max(1)) as f64;
            let gathered = center / denom;
            self.resources += gathered;
            return ActionDecision {
                move_one_step: false,
                move_signal: output.move_signal,
                direction_degrees,
                movement_spend: 0.0,
                gathered,
            };
        }

        let spend = output.move_signal.abs() * movement_cost_factor;
        let movement_spend = if spend.is_nan() {
            0.0
        } else {
            spend.min(self.resources)
        };
        self.resources -= movement_spend;
        ActionDecision {
            move_one_step: true,
            move_signal: output.move_signal,
            direction_degrees,
            movement_spend,
            gathered: 0.0,
        }
    }

    pub fn act(
        &mut self,
        left: f64,
        center: f64,
        right: f64,
        local_population: i32,
        movement_cost_factor: f64,
        _rng: &mut impl Rng,
        hidden_buf: &mut [f64],
        out2_buf: &mut [f64],
    ) -> f64 {
        let decision = self.decide_action(
            [left, left, center, right, right, center],
            local_population,
            movement_cost_factor,
            _rng,
            hidden_buf,
            out2_buf,
        );
        if !decision.move_one_step {
            return 0.0;
        }
        let dir = decision.direction_degrees.rem_euclid(360.0);
        let sign = if dir < 180.0 { 1.0 } else { -1.0 };
        decision.movement_spend * sign
    }

    pub fn reproduce(
        &mut self,
        rng: &mut impl Rng,
        next_id: &mut i32,
        reproduction_threshold: f64,
        reproduction_cost: f64,
        cfg: &Config,
    ) -> Option<ReproductionResult> {
        if self.resources < reproduction_threshold {
            return None;
        }
        self.resources -= reproduction_cost;
        let (child_brain, structural_mutation) = reproduce_brain(&self.brain, cfg, rng);
        let genetic_distance = self.brain.genetic_distance(&child_brain);
        let mut child = Individual::new(child_brain, self.position, *next_id);
        child.lineage_color = self.lineage_color;
        child.parent_id = Some(self.id);
        *next_id += 1;
        Some(ReproductionResult {
            child,
            structural_mutation,
            genetic_distance,
        })
    }

    /// Backward-compatible reproduce without config (uses defaults).
    pub fn reproduce_legacy(
        &mut self,
        rng: &mut impl Rng,
        next_id: &mut i32,
        reproduction_threshold: f64,
        reproduction_cost: f64,
        mutation_rate: f64,
        mutation_sigma: f64,
    ) -> Option<Individual> {
        let cfg = Config {
            reproduction_threshold,
            reproduction_cost,
            mutation_rate,
            mutation_sigma,
            ..Config::default()
        };
        self.reproduce(rng, next_id, reproduction_threshold, reproduction_cost, &cfg)
            .map(|r| r.child)
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "id={} pos={} res={:.3} age={} brain={:?} complexity={}",
            self.id,
            self.position,
            self.resources,
            self.age,
            self.brain.kind(),
            self.complexity()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{BrainInner, PerceptronBrain};
    use crate::config::ArchitecturePolicy;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_buffers(nodes: usize) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0; nodes.max(1)], vec![0.0; nodes.max(1)])
    }

    fn founder_perceptron() -> BrainGenome {
        BrainGenome {
            kind: crate::brain::BrainKind::Perceptron,
            inner: BrainInner::Perceptron(PerceptronBrain::new(8, 1)),
        }
    }

    #[test]
    fn decide_action_gathers_when_move_signal_nonpositive() {
        let mut brain = founder_perceptron();
        if let BrainInner::Perceptron(ref mut p) = brain.inner {
            for w in p.weights.iter_mut() {
                *w = -1.0;
            }
        }
        let mut ind = Individual::new(brain, 0, 1);
        ind.resources = 5.0;

        let mut rng = StdRng::seed_from_u64(1);
        let (mut hidden, mut out2) = make_buffers(4);
        let decision = ind.decide_action(
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            2,
            0.5,
            &mut rng,
            &mut hidden,
            &mut out2,
        );

        assert!(!decision.move_one_step);
        assert!(decision.gathered > 0.0);
    }

    #[test]
    fn structural_mutation_can_grow_mlp() {
        let mut rng = StdRng::seed_from_u64(99);
        let cfg = Config {
            architecture_policy: ArchitecturePolicy::EvolveMlp,
            architecture_mutation_rate: 1.0,
            nn_nodes: 1,
            nn_layers: 1,
            ..Config::default()
        };
        let parent = BrainGenome::new_founder(&cfg);
        assert_eq!(parent.kind, crate::brain::BrainKind::Mlp);
        let (child, mutation) = reproduce_brain(&parent, &cfg, &mut rng);
        assert!(matches!(
            mutation,
            StructuralMutation::MlpAddNodeLayer1
                | StructuralMutation::MlpAddLayer
                | StructuralMutation::ParametricOnly
        ));
        let _ = child;
    }

    #[test]
    fn founder_kind_matches_architecture_policy() {
        let perceptron = Config {
            architecture_policy: ArchitecturePolicy::PerceptronOnly,
            nn_nodes: 1,
            ..Config::default()
        };
        assert_eq!(
            BrainGenome::new_founder(&perceptron).kind,
            crate::brain::BrainKind::Perceptron
        );

        let mlp = Config {
            architecture_policy: ArchitecturePolicy::EvolveMlp,
            nn_nodes: 1,
            nn_layers: 1,
            ..Config::default()
        };
        assert_eq!(
            BrainGenome::new_founder(&mlp).kind,
            crate::brain::BrainKind::Mlp
        );

        let spatial_legacy = Config {
            architecture_policy: ArchitecturePolicy::EvolveSpatialGraph,
            spatial_blob_layout: "legacy_graph".to_string(),
            nn_nodes: 1,
            ..Config::default()
        };
        assert_eq!(
            BrainGenome::new_founder(&spatial_legacy).kind,
            crate::brain::BrainKind::SpatialGraph
        );

        let spatial_blob = Config {
            architecture_policy: ArchitecturePolicy::EvolveSpatialGraph,
            nn_nodes: 1,
            ..Config::default()
        };
        assert_eq!(
            BrainGenome::new_founder(&spatial_blob).kind,
            crate::brain::BrainKind::SpatialBlob
        );

        let goo = Config {
            architecture_policy: ArchitecturePolicy::EvolveGoo,
            ..Config::default()
        };
        assert_eq!(
            BrainGenome::new_founder(&goo).kind,
            crate::brain::BrainKind::GooGraph
        );
    }
}
