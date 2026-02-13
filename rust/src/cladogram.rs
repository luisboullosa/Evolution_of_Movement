/// Cladogram tracking and genetic distance calculation
use crate::individual::Individual;
use crate::neural_network::NeuralNetwork;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LineageNode {
    pub id: i32,
    pub parent_id: Option<i32>,
    pub children: Vec<i32>,
    pub birth_time: i32,
    pub death_time: Option<i32>,
    pub color: [u8; 3],
}

#[derive(Debug, Clone)]
pub struct Cladogram {
    pub nodes: HashMap<i32, LineageNode>,
    pub current_time: i32,
}

impl Cladogram {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            current_time: 0,
        }
    }

    pub fn add_founder(&mut self, id: i32, color: [u8; 3]) {
        self.nodes.insert(
            id,
            LineageNode {
                id,
                parent_id: None,
                children: Vec::new(),
                birth_time: self.current_time,
                death_time: None,
                color,
            },
        );
    }

    pub fn add_offspring(&mut self, parent_id: i32, child_id: i32, color: [u8; 3]) {
        // Add child node
        self.nodes.insert(
            child_id,
            LineageNode {
                id: child_id,
                parent_id: Some(parent_id),
                children: Vec::new(),
                birth_time: self.current_time,
                death_time: None,
                color,
            },
        );

        // Update parent's children list
        if let Some(parent) = self.nodes.get_mut(&parent_id) {
            parent.children.push(child_id);
        }
    }

    pub fn mark_death(&mut self, id: i32) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.death_time = Some(self.current_time);
        }
    }

    pub fn advance_time(&mut self) {
        self.current_time += 1;
    }

    /// Calculate genetic distance between two individuals based on neural network weights
    pub fn genetic_distance(nn1: &NeuralNetwork, nn2: &NeuralNetwork) -> f64 {
        let mut sum_sq_diff = 0.0;
        let mut count = 0;

        // Compare weights
        for (w1, w2) in nn1.weights.iter().zip(nn2.weights.iter()) {
            sum_sq_diff += (w1 - w2).powi(2);
            count += 1;
        }

        // Compare weights_multi
        for (w1, w2) in nn1.weights_multi.iter().zip(nn2.weights_multi.iter()) {
            sum_sq_diff += (w1 - w2).powi(2);
            count += 1;
        }

        // Compare biases
        for (b1, b2) in nn1.biases.iter().zip(nn2.biases.iter()) {
            sum_sq_diff += (b1 - b2).powi(2);
            count += 1;
        }

        // Compare weights2
        for (w1, w2) in nn1.weights2.iter().zip(nn2.weights2.iter()) {
            sum_sq_diff += (w1 - w2).powi(2);
            count += 1;
        }

        // Compare biases2
        for (b1, b2) in nn1.biases2.iter().zip(nn2.biases2.iter()) {
            sum_sq_diff += (b1 - b2).powi(2);
            count += 1;
        }

        if count > 0 {
            (sum_sq_diff / count as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Get all living descendants of a given node
    pub fn get_living_descendants(&self, id: i32) -> Vec<i32> {
        let mut descendants = Vec::new();
        let mut to_visit = vec![id];

        while let Some(current_id) = to_visit.pop() {
            if let Some(node) = self.nodes.get(&current_id) {
                if node.death_time.is_none() {
                    descendants.push(current_id);
                }
                for &child_id in &node.children {
                    to_visit.push(child_id);
                }
            }
        }

        descendants
    }

    /// Get the lineage path from a node to its root founder
    pub fn get_lineage_path(&self, id: i32) -> Vec<i32> {
        let mut path = Vec::new();
        let mut current_id = id;

        path.push(current_id);
        while let Some(node) = self.nodes.get(&current_id) {
            if let Some(parent_id) = node.parent_id {
                path.push(parent_id);
                current_id = parent_id;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cladogram_basic() {
        let mut clado = Cladogram::new();
        clado.add_founder(1, [255, 0, 0]);
        clado.add_offspring(1, 2, [255, 0, 0]);
        
        assert_eq!(clado.nodes.len(), 2);
        assert_eq!(clado.nodes[&1].children.len(), 1);
    }

    #[test]
    fn test_genetic_distance() {
        let nn1 = NeuralNetwork::new(2, 1, 1, 0);
        let mut nn2 = NeuralNetwork::new(2, 1, 1, 0);
        
        // Same networks should have distance 0
        let dist = Cladogram::genetic_distance(&nn1, &nn2);
        assert!(dist < 1e-10);

        // Different networks should have positive distance
        nn2.weights[0] = 1.0;
        let dist2 = Cladogram::genetic_distance(&nn1, &nn2);
        assert!(dist2 > 0.0);
    }
}
