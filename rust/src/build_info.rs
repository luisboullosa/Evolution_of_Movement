//! Compile-time build identity shared by all binaries.

/// Crate semver from Cargo.toml.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git short hash or build timestamp from build.rs.
pub const BUILD_ID: &str = env!("BUILD_ID");

/// CUDA architecture used when compiling the brain step kernel (e.g. `sm_89`).
pub const CUDA_ARCH: &str = env!("CUDA_ARCH");

/// Human-readable label, e.g. `v0.1.0 (a1b2c3d)`.
pub const VERSION_LABEL: &str = concat!(
    "v",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("BUILD_ID"),
    ")"
);
