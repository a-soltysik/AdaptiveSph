# AdaptiveSph

Experimental CUDA implementation of a weakly compressible smoothed particle hydrodynamics (WCSPH) fluid solver. It uses a uniform grid for particle-neighbor search, supports adaptive particle splitting and merging, and renders particles with Vulkan.

## Build

Requirements: Windows 10+ or Linux, CMake 3.25+, Ninja, an NVIDIA CUDA Toolkit with a supported host compiler, the Vulkan SDK, and a GPU/driver supporting CUDA–Vulkan external memory and semaphores. CMake downloads missing dependencies during configuration.

Choose a preset for your platform:

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel
```

```sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release --parallel
```

The executable is written to `build/<preset>/bin/`. Linux users can replace `linux-gcc-release` with `linux-clang-release` to use Clang.

## Run

Run from the repository root and pass one of the example JSON configurations to the executable:

| Configuration | Use |
| --- | --- |
| [configs/base.json](configs/base.json) | 13,754 particles; fixed resolution; 75% vertical fill. |
| [configs/adaptive.json](configs/adaptive.json) | 13,754 particles; adaptive splitting and merging; 75% vertical fill. |
| [configs/poiseuille.json](configs/poiseuille.json) | 15,341 particles; periodic channel flow; full vertical fill. |

```powershell
.\build\windows-msvc-release\bin\AdaptiveSph.exe configs\adaptive.json
```

```sh
./build/linux-gcc-release/bin/AdaptiveSph configs/adaptive.json
```

Configuration paths are relative to the current working directory. Without an argument, the application looks for `configs/base.json` beside the executable. See the [configuration guide](configs/README.md) for parameter meanings and experiment ideas. When refinement is enabled, the GUI can also request a one-shot refinement pass.

## Status and references

This is an experimental project. Numerical validation of adaptive refinement and the Poiseuille example is pending, and Linux GPU/Vulkan runtime has not been verified. The renderer is for visual inspection, not performance benchmarking.

Background papers are listed in [doc/REFERENCES.md](doc/REFERENCES.md). The project implementation is not claimed to reproduce their methods or results exactly.
