use crate::neural_network::NeuralNetwork;
use rand::Rng;
use rand_distr::{Cauchy, Distribution};

#[derive(Clone, Debug)]
pub struct Individual {
    pub nn: NeuralNetwork,
    pub resources: f64,
    pub position: usize,
    pub id: i32,
    pub age: i32,
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
        // Build inputs on the stack to avoid heap allocation: left, center, right, occupancy, internal resources
        let mut stack_inputs = [0.0f64; 5];
        stack_inputs[0] = left;
        stack_inputs[1] = center;
        stack_inputs[2] = right;
        stack_inputs[3] = local_population as f64;
        stack_inputs[4] = self.resources;

        // Call appropriate NN forward depending on NN input shape
        // Allow benchmarking the impact of buffer reuse by disabling via env var.
        // Set `DISABLE_NN_BUFFERS=1` to force the allocator-allocating codepaths.
        let disable_buf = std::env::var("DISABLE_NN_BUFFERS")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        let (direction, magnitude) = if disable_buf {
            if self.nn.inputs <= 1 {
                self.nn.forward_single(center)
            } else {
                self.nn.forward_multi(&stack_inputs)
            }
        } else {
            if self.nn.inputs <= 1 {
                // historically, single-input nets receive the local resource
                self.nn.forward_single_buf(center, hidden_buf, out2_buf)
            } else {
                // pass a slice view of the stack array; forward_multi will only read up to `self.inputs`
                self.nn
                    .forward_multi_buf(&stack_inputs, hidden_buf, out2_buf)
            }
        };

        // Decision: gather if magnitude < 0.5, otherwise move and spend
        if magnitude < 0.5 {
            let denom = (local_population.max(1)) as f64;
            let gathered = center / denom;
            self.resources += gathered;
            0.0
        } else {
            let spend = magnitude * movement_cost_factor;
            let spend_clamped = if spend.is_nan() {
                0.0
            } else {
                spend.min(self.resources)
            };
            self.resources -= spend_clamped;
            // return signed delta in C++ the direction determines sign; here we round later in Simulation
            // Use direction to determine movement sign: negative -> left, positive -> right
            if direction.is_nan() {
                spend_clamped
            } else {
                // embed magnitude as spend (Simulation uses absolute value) and returns delta based on rounding
                spend_clamped * direction.signum()
            }
        }
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
            let child = Individual::new(child_nn, self.position, *next_id);
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
