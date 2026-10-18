# Configuration guide

The JSON files in this directory are complete examples. Run one from the repository root:

```sh
./build/linux-gcc-release/bin/AdaptiveSph configs/adaptive.json
```

Use the matching executable path for your build preset. An explicit config path is relative to the current working directory; without a path, the app looks for `configs/base.json` beside the executable. Copy an example before experimenting: the parser requires all fields of each present simulation, refinement and rendering object, so keep unrelated fields when editing. `scenarioParameters.type` and `cooldown` have defaults, but each selected scenario needs its matching parameter object.

## Example configurations

| File | Purpose |
| --- | --- |
| [`base.json`](base.json) | 13,754 particles; fixed resolution; 75% vertical fill. |
| [`adaptive.json`](adaptive.json) | 13,754 particles; adaptive splitting and merging using velocity thresholds; 75% vertical fill. |
| [`poiseuille.json`](poiseuille.json) | 15,341 particles; channel flow periodic in X, driven along X; full vertical fill. |

## Top-level sections

| Section | Required? | Meaning |
| --- | --- | --- |
| `initialParameters` | Yes | Dimensions of the initial regular particle grid. Total fluid count is `x * y * z`. |
| `simulationParameters` | Yes | Domain, gravity, fluid properties, particle sizes and CUDA launch size. |
| `refinementParameters` | No; `null` disables refinement | Adaptive split/merge criteria, schedule and particle capacity. An object with `enabled: false` also selects fixed resolution. |
| `renderingParameters` | No | Display options; `renderBoundaryParticles` controls visibility of boundary particles. |
| `scenarioParameters` | No | Optional delayed scenario such as Poiseuille flow, a driven cavity or dam break. |

## Initial grid and simulation parameters

Physical quantities use SI units; counts and ratios are unitless. Each axis of `domain.min` and `domain.max` defines the simulation bounds. Initial particle centers have a 0.15 m margin from the walls. X and Z use the full usable span; Y is anchored at the `max.y` side and its occupied height is controlled by `verticalFillRatio`. The ratio targets the fraction of the domain height covered by particle diameters, subject to the wall margin. With `1.0`, the grid uses the full usable Y span.

| Parameter | Meaning and notes |
| --- | --- |
| `initialParameters.particleCount` | Number of initial fluid particles along each axis. Increasing the product increases startup and per-step work. |
| `initialParameters.verticalFillRatio` | Vertical fill fraction in `(0, 1]`. `0.75` leaves a quarter-height free surface; `1.0` fills the usable vertical span. The Y layer is placed toward `max.y`. Older configs without this field default to `1.0`. |
| `simulationParameters.domain.min`, `max` | Lower and upper domain corners in metres. Each extent must be greater than 0.30 m. |
| `simulationParameters.domain.friction` | Boundary friction coefficient. Set to zero to remove the wall-friction term; this is separate from `viscosityConstant`. |
| `simulationParameters.gravity` | Gravitational acceleration vector in m/s². |
| `simulationParameters.speedOfSound` | Sound speed used by the weakly-compressible pressure equation, in m/s. Changing it changes the pressure response. |
| `simulationParameters.restDensity` | Reference fluid density in kg/m³. |
| `simulationParameters.viscosityConstant` | Fluid kinematic-viscosity parameter in m²/s. Zero skips the fluid viscosity term. |
| `simulationParameters.surfaceTensionConstant` | Coefficient for the surface-tension term. Zero disables that term. |
| `simulationParameters.maxVelocity` | Upper bound on particle speed in m/s; velocities above it are clamped. A low value can hide an unstable setup. |
| `simulationParameters.baseParticleRadius` | Initial particle/render radius in metres. Boundary-particle spacing is twice this value. |
| `simulationParameters.baseSmoothingRadius` | Initial smoothing length `h` in metres. The Wendland kernel has support out to `2h`. |
| `simulationParameters.baseParticleMass` | Mass of an initial fluid particle in kg. For a 3-D grid, use approximately `restDensity * dx * dy * dz`; the app warns when this differs by more than 10%. |
| `simulationParameters.threadsPerBlock` | CUDA threads per block, validated in `[1, 1024]`. Leave at 256 unless tuning and profiling on the target GPU. |

The base and adaptive examples use 0.05 m pitch in X/Z and 0.049 m in Y, so `baseParticleMass: 0.1225` matches `restDensity * dx * dy * dz` (`1000 * 0.05 * 0.049 * 0.05`). Poiseuille uses 0.05 m in all axes and a 0.125 kg base mass. If you change the domain, fill ratio or grid count, recheck the pitch and mass together.

The simulation step is currently fixed at `0.0001` s in the application and is not a JSON parameter. Refinement cooldowns below are measured in simulation steps; scenario cooldowns are measured in simulated seconds.

## Adaptive refinement

`refinementParameters.enabled` turns adaptation on. Each split keeps a parent particle at the center and adds 12 children arranged on an icosahedron. Merging combines a nearby compatible pair. Refinement runs automatically according to its step schedule; the GUI can request an extra pass on the next simulation step.

| Parameter | Meaning |
| --- | --- |
| `criterionType` | One shared split/merge metric: `velocity`, `vorticity` or `interface` (see below). |
| `initialCooldown` | Step index of the first scheduled refinement pass. |
| `cooldown` | Must be greater than zero. Later passes run when the step index is a multiple of this value. |
| `maxParticleCount` | Hard fluid-particle capacity allocated for the adaptive simulation. It must be at least the initial count; larger capacities use more GPU memory even if the run never reaches them. |
| `maxBatchRatio` | Maximum fraction of the current particle count selected as split parents or merge candidates per pass, rounded to an integer. Splitting each selected parent adds 12 particles, subject to the capacity limit. Valid range: `(0, 1]`. |
| `minMassRatio` | A particle must have at least this multiple of `baseParticleMass` to be considered for splitting. This prevents already-small particles from being split again. |
| `maxMassRatio` | Particles above this multiple of `baseParticleMass` are not merge candidates; a merged pair must also stay below this mass limit. |
| `splitting.epsilon` | Child offset as a multiple of the parent's smoothing length. Larger values spread children farther from the parent. |
| `splitting.alpha` | New smoothing length as a fraction of the parent's smoothing length. |
| `splitting.centerMassRatio` | Fraction of the parent's mass retained by the center particle. |
| `splitting.vertexMassRatio` | Fraction assigned to each of the 12 children. `centerMassRatio + 12 * vertexMassRatio` must equal 1. |

Criterion thresholds are in the nested parameter objects:

| `criterionType` | Split condition | Merge condition |
| --- | --- | --- |
| `velocity` | Speed magnitude exceeds `velocity.split.minimalSpeedThreshold` (m/s). | Speed magnitude is below `velocity.merge.maximalSpeedThreshold` (m/s). |
| `vorticity` | Vorticity magnitude exceeds `vorticity.split.minimalVorticityThreshold` (s⁻¹). | Vorticity magnitude is below `vorticity.merge.maximalVorticityThreshold` (s⁻¹). At least four valid fluid neighbors are needed to evaluate it. |
| `interface` | Magnitude of the kernel-weighted mean neighbor displacement exceeds `interfaceParameters.split.proximityMagnitudeThreshold` (m). | It is below `interfaceParameters.merge.proximityMagnitudeThreshold` (m). |

Only particles inside the mass limits can pass a criterion. Candidate selection is capped by `maxBatchRatio`; available capacity can further reduce the number of splits. For predictable comparisons, keep the initial grid, mass, capacity and schedule fixed while comparing criteria or thresholds.

## Scenarios

`scenarioParameters.type` accepts `none`, `poiseuille_flow`, `lid_driven_cavity` or `dam_break`. `cooldown` delays scenario activation in simulated seconds (default 2.5 s). The scenario-specific object required by the selected type is:

- `poiseuille_flow`: `poiseuilleFlow.drivingForce` adds acceleration in X after activation. Selecting this type makes the X domain periodic from startup; the Poiseuille example also uses zero gravity.
- `lid_driven_cavity`: a positive `lidDrivenCavity.lidVelocity` sets the moving boundary speed along X after activation.
- `dam_break`: `damBreak.expandedDomain` gives the domain bounds and friction used when the scenario expands the domain after activation.

## Experiments

- **Compare refinement criteria:** copy `adaptive.json`, keep the same scene and capacity, then compare `velocity`, `vorticity` and `interface`. Tune only the corresponding split/merge thresholds between runs.
- **Control adaptation rate:** change `cooldown` or `maxBatchRatio` while keeping thresholds fixed. Inspect particle counts and refinement-pass statistics in the GUI/log.
- **Study capacity pressure:** vary `maxParticleCount` and observe when it limits splitting. Remember that capacity is reserved up front, so avoid raising it without checking GPU memory.
- **Change spatial resolution or fill level:** alter `particleCount`, `verticalFillRatio` and/or the domain, then recalculate grid pitch and `baseParticleMass`. Keep the initial mass consistent with `restDensity * dx * dy * dz` when comparing density behavior.
- **Compare forcing:** for the Poiseuille case, vary `drivingForce` while holding viscosity and geometry fixed; for a gravity-driven scene, change only `gravity` first.

These runs are exploratory. Adaptive refinement and the Poiseuille flow have not yet been numerically validated, so visual output alone does not establish accuracy.
