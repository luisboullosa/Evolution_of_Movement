use crate::environment::Environment;
use crate::individual::Individual;
use crate::config::Config;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct Simulation {
    pub env: Environment,
    pub individuals: Vec<Individual>,
    pub rng: StdRng,
    pub next_id: i32,
    pub last_spend: Vec<f64>,
    pub last_delta: Vec<i32>,
    pub last_movers: usize,
    pub config: Config,
}

impl Simulation {
    pub fn new(cfg: &Config) -> Self {
        let mut env = Environment::new(cfg.grid_width);
        env.init_perlin(cfg.seed as u64);
        Simulation {
            env,
            individuals: Vec::new(),
            rng: StdRng::seed_from_u64(cfg.seed as u64),
            next_id: 1,
            last_spend: Vec::new(),
            last_delta: Vec::new(),
            last_movers: 0,
            config: cfg.clone(),
        }
    }

    pub fn init_population(&mut self, n: usize) {
        for _ in 0..n {
            let nn = crate::neural_network::NeuralNetwork::new(self.config.nn_nodes as usize, self.config.nn_inputs as usize, self.config.nn_layers, self.config.nn_nodes2 as usize);
            let ind = Individual::new(nn, 0, self.next_id);
            self.next_id += 1;
            self.individuals.push(ind);
        }
    }

    pub fn step(&mut self) {
        // simple step: advance environment and let each individual gather
        self.env.step(self.config.env_change_rate, self.config.seed as u64, self.config.env_use_perlin_time, self.config.env_time_speed, self.config.env_time_amplitude, self.config.env_perlin_spatial_scale, self.config.env_perlin_time_scale);
        for ind in &mut self.individuals {
            let pos = ind.position;
            let gathered = self.env.gather_at(pos, 1);
            ind.act(gathered, gathered, gathered, 1, &mut self.rng);
        }
    }

    pub fn run(&mut self) {
        for _ in 0..self.config.steps {
            self.step();
        }
    }
}
