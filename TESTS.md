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

Notes
- The `test` target builds the project and runs the tests via CTest.
- If you prefer to run tests without the CMake `test` target, run `ctest` directly from the `build` directory.
- Test binaries are built into the `build` directory alongside other targets.
