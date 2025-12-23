pub mod environment {
    pub struct Environment {
        // TODO: port fields from C++ `Environment`
    }

    impl Environment {
        pub fn new() -> Self {
            Environment {}
        }
    }
}

pub mod individual {
    pub struct Individual {
        // TODO: port fields from C++ `Individual`
    }

    impl Individual {
        pub fn new() -> Self {
            Individual {}
        }
    }
}

pub mod simulation {
    use crate::environment::Environment;

    pub struct Simulation {
        // TODO: port fields from C++ `Simulation`
        _env: Environment,
    }

    impl Simulation {
        pub fn new(env: Environment) -> Self {
            Simulation { _env: env }
        }
    }
}
