Running the project's tests
===========================

C++ tests (CMake / CTest)
-------------------------

This project registers a couple of unit tests with CTest.

**Linux / macOS** (from the project root):

```bash
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release
ctest --test-dir build/Release --output-on-failure
```

**Windows (PowerShell / MinGW)**:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Rust workspace

--------------
The Rust port has its own Cargo workspace under `rust/`. Use Cargo to build, lint and
run the Rust binary (works on all platforms):

```bash
cd rust
cargo clippy
cargo test
cargo build --release
cargo run -- --help
```

Notes

- On Linux install `gcc-c++` (Fedora/RHEL) or `build-essential` (Debian/Ubuntu) for the C++ build.
- The CMake `test` target builds the project and runs the tests via CTest.
- If you prefer to run tests without the CMake `test` target, run `ctest` directly from the `build` directory.
- Test binaries are built into the `build` directory alongside other targets.
