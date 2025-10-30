# evo-movement-sim

Minimal C++ simulation of evolving movement in digital individuals.

Build (Windows, PowerShell):

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022"; cmake --build . --config Release
```

Run (Release):

```powershell
.
# from build/Release or run the exe directly
./evo-movement-sim.exe
```

This project is intentionally small and dependency-free (uses only the C++ standard library).

Web viewer
----------
The simulator can write frames to `webviewer/frames.tsv`. After running the simulation (the default build target
writes to that path), open `webviewer/index.html` in your browser and load the `frames.tsv` file using the file picker.

Cleaning webviewer output
-------------------------
The simulator writes a few generated files into `webviewer/` (frames.tsv, timestamped `env.tsv.*.tsv`,
`individuals.tsv.*.tsv`, and `nn_index.json`). To make it easy to remove these files you can use the
provided cleaner script or a CMake custom target.

From PowerShell (dry-run):

```powershell
python scripts\clean_webviewer.py
```

To actually delete the discovered files add `--yes`:

```powershell
python scripts\clean_webviewer.py --yes
```

Alternatively, if you use the included CMake target you can run:

```powershell
# from the project root
cmake --build build --target clean-webviewer --config Release
```

The CMake target runs the cleaner in dry-run mode by default and prints which files would be removed; to actually delete files run the Python script with `--yes` as shown above.

Running with burn-in and steps
-----------------------------
You can set a burn-in period (timesteps to skip before recording frames/snapshots) and number of steps from the command line:

```powershell
# run 500 timesteps but only start recording after timestep 200
.
# from build/Release or run the exe directly
./evo-movement-sim.exe --burn-in=200 --steps=500 --config-out=webviewer\simNNNN\config.json
```

`--burn-in=N` causes the simulator to perform all timesteps but suppress console/frame/snapshot reporting until `t >= N`.



Running the tests
-----------------
Test instructions were moved to `TESTS.md`. See `TESTS.md` for commands to build and run the project's tests.

