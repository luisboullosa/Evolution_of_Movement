use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Try to get git commit short hash
    let mut build_id = String::new();
    if let Ok(output) = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                build_id = s.trim().to_string();
            }
        }
    }
    if build_id.is_empty() {
        // fallback to timestamp
        build_id = format!("ts-{}", chrono::offset::Utc::now().format("%Y%m%d%H%M%S"));
    }
    // expose to rustc env
    println!("cargo:rustc-env=BUILD_ID={}", build_id);
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    println!("cargo:rustc-env=CUDA_ARCH={}", cuda_arch);
    // also write to OUT_DIR for reference
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let _ = fs::write(out_dir.join("build_id.txt"), &build_id);

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        let cuda_src = PathBuf::from("cuda/brain_step_kernel.cu");
        let cuda_src_str = cuda_src
            .to_str()
            .expect("cuda/brain_step_kernel.cu must be valid UTF-8");
        println!("cargo:rerun-if-changed={}", cuda_src.display());
        println!("cargo:rerun-if-env-changed=CUDA_ARCH");
        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
        let kernel_ptx = out_dir.join("brain_step_kernel.ptx");
        let kernel_ptx_str = kernel_ptx
            .to_str()
            .expect("output path must be valid UTF-8");
        let nvcc_status = Command::new("nvcc")
            .args(&["--ptx", "-arch", &arch, "-o", kernel_ptx_str, cuda_src_str])
            .status()
            .expect("nvcc not found; install the CUDA toolkit to build the brain step kernel");
        if !nvcc_status.success() {
            panic!("nvcc failed to compile {} with arch {}", cuda_src_str, arch);
        }
        println!("cargo:rustc-env=TENSOR_KERNEL_PTX={}", kernel_ptx.display());
        let tensor_path_out = out_dir.join("tensor_kernel_path.rs");
        let path_literal = kernel_ptx.display();
        let path_code = format!(
            "pub const BUILT_TENSOR_KERNEL_PTX: &str = r#\"{}\"#;",
            path_literal
        );
        fs::write(&tensor_path_out, path_code)
            .expect("failed to write tensor kernel path helper");
    } else {
        let tensor_path_out = out_dir.join("tensor_kernel_path.rs");
        let path_code = "pub const BUILT_TENSOR_KERNEL_PTX: &str = \"\";";
        fs::write(&tensor_path_out, path_code)
            .expect("failed to write tensor kernel path helper");
    }
    println!("cargo:rerun-if-changed=assets/fitness_landscape_viewer.html");
}
