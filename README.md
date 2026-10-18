# AdaptiveSph

AdaptiveSph is an experimental weakly compressible SPH (WCSPH) fluid solver written in CUDA, with particle rendering through Vulkan. It explores neighbor search on a uniform grid and adaptive particle refinement; the refinement path and numerical behavior still need dedicated validation.

## What is implemented

- CUDA kernels for density, pressure, viscosity, boundary forces, and particle integration.
- Uniform-grid neighbor search with particles sorted by cell key.
- Optional particle splitting and merging, enabled by an adaptive configuration.
- A Vulkan renderer for fluid and boundary particles, with CUDA/Vulkan shared memory.
- A small ImGui overlay for simulation statistics and controls.

The renderer is for inspecting simulation behavior. Its frame rate is not a solver benchmark.

## Project layout

- `cuda/` contains the SPH kernels, neighbor grid, boundary handling, and refinement code.
- `app/` contains the application loop, configuration loading, input, and GUI.
- `engine/` contains the Vulkan rendering support used by the application.
- `configs/` contains the base, adaptive and Poiseuille example scenes.

### CUDA–Vulkan interop

Vulkan creates the shared particle buffers and two binary semaphores; CUDA imports them. CUDA writes particle data, and Vulkan reads it. Before writing the next frame, CUDA waits for `vulkanDone` from the previous render. After updating the particles, CUDA signals `cudaReady`; Vulkan waits for it before drawing and signals `vulkanDone` when the render finishes.

## Requirements

Both builds need CMake 3.25 or later, Ninja, an NVIDIA CUDA Toolkit, and a host compiler supported by that Toolkit. They also need the Vulkan SDK or development headers and a GPU/driver that supports both CUDA and the Vulkan external-memory path used by the renderer. CMake downloads project dependencies on the first configure if the dependency cache is empty.

- Windows: Windows 10 or later and an MSVC version supported by the selected CUDA Toolkit.
- Linux: GCC or Clang, plus the Vulkan development packages for the distribution.

The repository has Windows MSVC and Linux GCC/Clang presets. CUDA targets forward MSVC's `/Zc:preprocessor` option, which is required by the CUDA 13.4 CCCL headers. The Linux GCC preset with clang-tidy builds in WSL; Linux GPU/Vulkan runtime remains unverified.

## Build and run

Run commands from the repository root. The Windows executable is written to `build/windows-msvc-release/bin/AdaptiveSph.exe`; CMake copies the example configurations beside it for IDE launches.

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel
.\build\windows-msvc-release\bin\AdaptiveSph.exe configs\base.json
```

On Linux, the GCC Release preset writes `build/linux-gcc-release/bin/AdaptiveSph`:

```sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release --parallel
./build/linux-gcc-release/bin/AdaptiveSph configs/base.json
```

To use Clang on Linux, replace `linux-gcc-release` with `linux-clang-release` in both CMake commands and the executable path. CUDA must support the selected host compiler.

The base scene starts with 2,160 fluid particles. It uses a 0.1 m grid pitch, a 0.05 m particle radius, a 0.12 m smoothing radius, 1 kg particle mass, and 1,000 kg/m³ rest density. The mass matches rest density times the initial grid-cell volume. Coordinates, time, mass, density, and speed values use SI units in the example configurations.

The adaptive variant uses the same initial scene and physical parameters, with an 8,192 particle capacity. Run it with the matching executable:

```powershell
.\build\windows-msvc-release\bin\AdaptiveSph.exe configs\adaptive.json
```

```sh
./build/linux-gcc-release/bin/AdaptiveSph configs/adaptive.json
```

Refinement starts automatically after the configured initial cooldown in simulation steps. The GUI button advances that schedule.

For a channel with periodic X boundaries, zero gravity and a driving acceleration of 0.1 m/s² in X, use `configs/poiseuille.json`:

```powershell
.\build\windows-msvc-release\bin\AdaptiveSph.exe configs\poiseuille.json
```

```sh
./build/linux-gcc-release/bin/AdaptiveSph configs/poiseuille.json
```

The force is applied after the Poiseuille scenario activates. This config starts the scenario immediately and enables vorticity-based adaptive refinement after 100 simulation steps, then repeats refinement every 100 steps. Numerical validation of the flow remains pending.

Without a configuration argument, the application looks for `configs/base.json` beside the executable. Explicit configuration paths are relative to the current working directory. `AdaptiveSph --help` logs the command syntax. A missing or invalid configuration aborts startup; the application does not generate fallback parameters. Runs have no automatic duration limit.

## Controls

- Hold the left mouse button and move the mouse to look around.
- Use `W`, `A`, `S`, and `D` to move; use `Space` and `Left Shift` to move vertically.
- Use the GUI to inspect density and neighbor statistics, change the domain, or enable adaptive refinement.
