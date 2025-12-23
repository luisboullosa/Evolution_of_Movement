use crate::neural_network::NeuralNetwork;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Individual {
    pub nn: NeuralNetwork,
    pub resources: f64,
    pub position: usize,
    pub id: i32,
    pub age: i32,
}

impl Individual {
    pub const MAINTENANCE_COST_PER_NODE: f64 = 0.0015;
    pub const REPRODUCTION_THRESHOLD: f64 = 20.0;
    pub const REPRODUCTION_COST: f64 = 5.0;
    pub const MUTATION_RATE: f64 = 0.01;
    pub const MUTATION_SIGMA: f64 = 0.1;
    pub const MOVEMENT_COST_FACTOR: f64 = 2.0;

    pub fn new(nn: NeuralNetwork, pos: usize, id: i32) -> Self {
        Individual { nn, resources: 10.0, position: pos, id, age: 0 }
    }

    pub fn act(&mut self, left: f64, center: f64, right: f64, local_population: i32, rng: &mut impl Rng) -> f64 {
        // Placeholder: choose to gather at current pos; return movement cost 0.0
        self.resources += center;
        self.age += 1;
        0.0
    }

    pub fn reproduce(&mut self, rng: &mut impl Rng, next_id: &mut i32) -> Option<Individual> {
        if self.resources >= Self::REPRODUCTION_THRESHOLD {
            self.resources -= Self::REPRODUCTION_COST;
            let child_nn = self.nn.clone();
            let child = Individual::new(child_nn, self.position, *next_id);
            *next_id += 1;
            Some(child)
        } else {
            None
        }
    }
}
