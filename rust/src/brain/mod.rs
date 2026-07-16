pub mod genome;
pub mod goo_graph;
#[cfg(not(feature = "playground"))]
pub mod gpu_pack;
pub mod mlp;
pub mod mutation;
pub mod perceptron;
pub mod spatial_blob;
pub mod spatial_graph;

pub use genome::{BrainGenome, BrainInner, BrainKind, BrainOutput, ForwardContext, SpatialRuntime, PERCEPTRON_SENSOR_DIM, SENSOR_DIM, EMBED_DIM, THOUGHT_DIM};
pub use goo_graph::{GooGraphBrain, GooOp};
pub use mlp::MlpBrain;
pub use mutation::{
    apply_lab_structural, init_founder_weights, init_founder_weights_with_mode, reproduce_brain,
    FounderWeightMode, LabStructuralOp, StructuralMutation,
};
pub use perceptron::PerceptronBrain;
pub use spatial_blob::{SpatialBlobBrain, SpatialBlobLayout};
pub use spatial_graph::SpatialGraphBrain;

#[cfg(test)]
mod brain_tests {
    use super::genome::{BrainGenome, BrainInner, BrainKind, ForwardContext, SENSOR_DIM};
    use super::goo_graph::GooGraphBrain;
    use super::spatial_graph::SpatialGraphBrain;

    #[test]
    fn goo_forward_is_deterministic() {
        let brain = BrainGenome {
            kind: BrainKind::GooGraph,
            inner: BrainInner::GooGraph(GooGraphBrain::new_initial()),
        };
        let sensors = [1.0; SENSOR_DIM];
        let mut h1 = [0.0; 16];
        let mut h2 = [0.0; 16];
        let o1 = brain.forward(ForwardContext {
            sensors,
            position: 5,
            hidden_buf: &mut h1,
            out2_buf: &mut h2,
            spatial_runtime: None,
        });
        let o2 = brain.forward(ForwardContext {
            sensors,
            position: 5,
            hidden_buf: &mut h1,
            out2_buf: &mut h2,
            spatial_runtime: None,
        });
        assert!((o1.move_signal - o2.move_signal).abs() < 1e-12);
    }

    #[test]
    fn spatial_graph_updates_runtime() {
        let sg = SpatialGraphBrain::new(2);
        let brain = BrainGenome {
            kind: BrainKind::SpatialGraph,
            inner: BrainInner::SpatialGraph(sg),
        };
        let mut runtime = brain.init_spatial_runtime().unwrap();
        let sensors = [0.5; SENSOR_DIM];
        let mut hb = [0.0; 8];
        let mut ob = [0.0; 8];
        brain.forward(ForwardContext {
            sensors,
            position: 10,
            hidden_buf: &mut hb,
            out2_buf: &mut ob,
            spatial_runtime: Some(&mut runtime),
        });
        assert!(!runtime.node_embeddings.is_empty());
    }

    #[test]
    fn structure_signature_from_json_matches_brain() {
        let brains = [
            BrainGenome::new_founder(&crate::config::Config::default()),
            BrainGenome {
                kind: BrainKind::GooGraph,
                inner: BrainInner::GooGraph(GooGraphBrain::new_initial()),
            },
            BrainGenome {
                kind: BrainKind::SpatialGraph,
                inner: BrainInner::SpatialGraph(SpatialGraphBrain::new(3)),
            },
        ];
        for brain in &brains {
            let json = serde_json::to_string(brain).expect("serialize brain");
            let from_json =
                BrainGenome::structure_signature_from_brain_json(&json).expect("parse signature");
            assert_eq!(from_json, brain.structure_signature());
        }
    }
}
