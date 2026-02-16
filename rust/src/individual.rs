use crate::neural_network::NeuralNetwork;
use rand::Rng;
use rand_distr::{Distribution, Cauchy};

#[derive(Clone, Copy, Debug)]
pub struct ActionDecision {
    pub move_one_step: bool,
    pub move_signal: f64,
    pub direction_degrees: f64,
    pub movement_spend: f64,
    pub gathered: f64,
}

#[derive(Clone, Debug)]
pub struct Individual {
    pub nn: NeuralNetwork,
    pub resources: f64,
    pub position: usize,
    pub id: i32,
    pub age: i32,
    pub lineage_color: [u8; 3], // RGB color inherited from founder
    pub parent_id: Option<i32>,  // Track parent for cladogram
}

impl Individual {
    pub const MAINTENANCE_COST_PER_NODE: f64 = 0.01;
    pub const REPRODUCTION_THRESHOLD: f64 = 20.0;
    pub const REPRODUCTION_COST: f64 = 5.0;
    pub const MUTATION_RATE: f64 = 0.01;
    pub const MUTATION_SIGMA: f64 = 0.1;
    pub const MOVEMENT_COST_FACTOR: f64 = 0.1;

    pub fn new(nn: NeuralNetwork, pos: usize, id: i32) -> Self {
        Individual {
            nn,
            resources: 10.0,
            position: pos,
            id,
            age: 0,
            lineage_color: [128, 128, 128], // default gray
            parent_id: None,
        }
    }

    pub fn new_with_color(nn: NeuralNetwork, pos: usize, id: i32, color: [u8; 3]) -> Self {
        Individual {
            nn,
            resources: 10.0,
            position: pos,
            id,
            age: 0,
            lineage_color: color,
            parent_id: None,
        }
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
        let center = edge_resources.iter().sum::<f64>() / 6.0;
        let mut stack_inputs = [0.0f64; 8];
        stack_inputs[0] = edge_resources[0];
        stack_inputs[1] = edge_resources[1];
        stack_inputs[2] = edge_resources[2];
        stack_inputs[3] = edge_resources[3];
        stack_inputs[4] = edge_resources[4];
        stack_inputs[5] = edge_resources[5];
        stack_inputs[6] = local_population as f64;
        stack_inputs[7] = self.resources;

        let disable_buf = std::env::var("DISABLE_NN_BUFFERS")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        let (move_signal, direction_signal) = if disable_buf {
            if self.nn.inputs <= 1 {
                self.nn.forward_single(center)
            } else {
                self.nn.forward_multi(&stack_inputs)
            }
        } else if self.nn.inputs <= 1 {
            self.nn.forward_single_buf(center, hidden_buf, out2_buf)
        } else {
            self.nn.forward_multi_buf(&stack_inputs, hidden_buf, out2_buf)
        };

        let direction_degrees = Self::direction_signal_to_degrees(direction_signal);
        if move_signal <= 0.0 {
            let denom = (local_population.max(1)) as f64;
            let gathered = center / denom;
            self.resources += gathered;
            return ActionDecision {
                move_one_step: false,
                move_signal,
                direction_degrees,
                movement_spend: 0.0,
                gathered,
            };
        }

        let spend = move_signal.abs() * movement_cost_factor;
        let movement_spend = if spend.is_nan() {
            0.0
        } else {
            spend.min(self.resources)
        };
        self.resources -= movement_spend;
        ActionDecision {
            move_one_step: true,
            move_signal,
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
        mutation_rate: f64,
        mutation_sigma: f64,
    ) -> Option<Individual> {
        if self.resources >= reproduction_threshold {
            self.resources -= reproduction_cost;
            let mut child_nn = self.nn.clone();
            let cauchy = Cauchy::new(0.0, mutation_sigma).unwrap();
            for w in child_nn.weights.iter_mut() {
                if rng.gen::<f64>() < mutation_rate {
                    *w += cauchy.sample(rng);
                }
            }
            for w in child_nn.weights_multi.iter_mut() {
                if rng.gen::<f64>() < mutation_rate {
                    *w += cauchy.sample(rng);
                }
            }
            for b in child_nn.biases.iter_mut() {
                if rng.gen::<f64>() < mutation_rate {
                    *b += cauchy.sample(rng);
                }
            }
            for w in child_nn.weights2.iter_mut() {
                if rng.gen::<f64>() < mutation_rate {
                    *w += cauchy.sample(rng);
                }
            }
            for b in child_nn.biases2.iter_mut() {
                if rng.gen::<f64>() < mutation_rate {
                    *b += cauchy.sample(rng);
                }
            }
            let child = Individual {
                nn: child_nn,
                resources: 10.0,
                position: self.position,
                id: *next_id,
                age: 0,
                lineage_color: self.lineage_color, // inherit color
                parent_id: Some(self.id),          // track parent
            };
            *next_id += 1;
            Some(child)
        } else {
            None
        }
    }

    /// Short textual summary useful during debugging.
    pub fn debug_summary(&self) -> String {
        let (nodes, inputs, nodes2, layers) = self.nn.shape();
        format!(
            "id={} pos={} res={:.3} age={} nn=[nodes={},inputs={},layers={},nodes2={}]",
            self.id, self.position, self.resources, self.age, nodes, inputs, layers, nodes2
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_buffers(nodes: usize, nodes2: usize) -> (Vec<f64>, Vec<f64>) {
        let hidden = vec![0.0; nodes.max(1)];
        let out2 = vec![0.0; nodes2.max(1)];
        (hidden, out2)
    }

    #[test]
    fn decide_action_gathers_when_move_signal_nonpositive() {
        let mut nn = NeuralNetwork::new(1, 1, 1, 0);
        nn.weights[0] = -1.0;
        nn.biases[0] = 0.0;

        let mut ind = Individual::new(nn, 0, 1);
        ind.resources = 5.0;

        let mut rng = StdRng::seed_from_u64(1);
        let (mut hidden, mut out2) = make_buffers(ind.nn.nodes, ind.nn.nodes2);
        let decision = ind.decide_action([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 2, 0.5, &mut rng, &mut hidden, &mut out2);

        assert!(!decision.move_one_step);
        assert!(decision.move_signal <= 0.0);
        assert!(decision.gathered > 0.0);
        assert!(decision.movement_spend.abs() < 1e-12);
        assert!(ind.resources > 5.0);
    }

    #[test]
    fn decide_action_moves_and_spends_when_move_signal_positive() {
        let mut nn = NeuralNetwork::new(1, 1, 1, 0);
        nn.weights[0] = 0.0;
        nn.biases[0] = 2.0;

        let mut ind = Individual::new(nn, 0, 2);
        ind.resources = 5.0;

        let mut rng = StdRng::seed_from_u64(2);
        let (mut hidden, mut out2) = make_buffers(ind.nn.nodes, ind.nn.nodes2);
        let before = ind.resources;
        let decision = ind.decide_action([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1, 0.5, &mut rng, &mut hidden, &mut out2);

        assert!(decision.move_one_step);
        assert!(decision.move_signal > 0.0);
        assert!(decision.movement_spend > 0.0);
        assert!(decision.gathered.abs() < 1e-12);
        assert!(ind.resources < before);
    }

    #[test]
    fn decide_action_uses_second_output_for_direction_degrees() {
        let mut nn = NeuralNetwork::new(2, 1, 1, 0);
        nn.weights[0] = 0.0;
        nn.biases[0] = 2.0;
        nn.weights[1] = 1.0;
        nn.biases[1] = 0.0;

        let mut ind_low = Individual::new(nn.clone(), 0, 3);
        let mut ind_high = Individual::new(nn, 0, 4);
        let mut rng = StdRng::seed_from_u64(3);

        let (mut hidden_l, mut out2_l) = make_buffers(ind_low.nn.nodes, ind_low.nn.nodes2);
        let low = ind_low.decide_action([0.0, 0.0, -10.0, 0.0, 0.0, 0.0], 1, 0.5, &mut rng, &mut hidden_l, &mut out2_l);

        let (mut hidden_h, mut out2_h) = make_buffers(ind_high.nn.nodes, ind_high.nn.nodes2);
        let high = ind_high.decide_action([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], 1, 0.5, &mut rng, &mut hidden_h, &mut out2_h);

        assert!(low.direction_degrees >= 0.0 && low.direction_degrees <= 360.0);
        assert!(high.direction_degrees >= 0.0 && high.direction_degrees <= 360.0);
        assert!(high.direction_degrees > low.direction_degrees);
    }
}
