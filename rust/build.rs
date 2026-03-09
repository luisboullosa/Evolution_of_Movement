use std::process::Command;
use std::env;
use std::fs;

fn main() {
    // Try to get git commit short hash
    let mut build_id = String::new();
    if let Ok(output) = Command::new("git").args(["rev-parse", "--short", "HEAD"]).output() {
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
    // also write to OUT_DIR for reference
    if let Ok(outdir) = env::var("OUT_DIR") {
        let _ = fs::write(format!("{}/build_id.txt", outdir), &build_id);
    }
}
