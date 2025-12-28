Running the project's tests
===========================

This project registers a couple of unit tests with CTest. You can build and run them from the project root using PowerShell.

From PowerShell (project root):

```powershell
cmake -S . -B build
cmake --build build --config Release
cmake --build build --target test --config Release
# Alternatively, run ctest from the build directory:
# cd build; ctest -C Release --output-on-failure
```

Rust workspace
--------------
The Rust port has its own Cargo workspace under `rust/`. Use Cargo to build, lint and
run the Rust binary:

```powershell
pushd rust
cargo clippy
cargo build --release
cargo run -- --help
popd
```

Notes
- The `test` target builds the project and runs the tests via CTest.
- If you prefer to run tests without the CMake `test` target, run `ctest` directly from the `build` directory.
- Test binaries are built into the `build` directory alongside other targets.
