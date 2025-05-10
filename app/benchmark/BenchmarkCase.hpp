// BenchmarkFramework.hpp
/*#pragma once

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <fstream>
#include <glm/glm.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "panda/Logger.h"

namespace sph
{

struct BenchmarkParameters
{
    bool enabled = false;
    std::string testCase = "lidDrivenCavity";
    std::string outputPath = "benchmarks/";

    struct SimulationConfig
    {
        float particleSize = 0.05f;
    };

    struct AdaptiveConfig
    {
        float minParticleSize = 0.025f;
        float maxParticleSize = 0.05f;
    };

    SimulationConfig coarse;
    SimulationConfig fine;
    AdaptiveConfig adaptive;

    uint32_t measurementInterval = 20;
    uint32_t totalSimulationFrames = 1000;

    // Test case specific parameters
    float reynoldsNumber = 100.0f;
};

struct PerformanceMetrics
{
    double simulationTimeMs;  // Per frame computation time
    uint32_t particleCount;   // Current particle count
    double frameTimeMs;       // Total frame time
    double memoryUsageBytes;  // CUDA memory usage
};

struct ConservationMetrics
{
    float totalMass;
    float totalKineticEnergy;
    float totalMomentum[3];
};

struct BenchmarkResults
{
    // Common metrics
    std::vector<float> timeSteps;
    std::vector<PerformanceMetrics> performanceMetrics;
    std::vector<ConservationMetrics> conservationMetrics;

    // Velocity/density field errors
    std::vector<float> velocityErrors;
    std::vector<float> densityErrors;

    // Test case specific metrics
    std::vector<glm::vec2> vxProfile;  // For lid-driven cavity: (y, vx)
    std::vector<glm::vec2> vyProfile;  // For lid-driven cavity: (x, vy)
    float vortexCenterX;               // For lid-driven cavity: vortex center
    float vortexCenterY;
    float kineticEnergy;

    float leadingEdgePosition;        // For dam break: front position
    std::vector<float> gaugeHeights;  // For dam break: water heights at gauges

    float dragCoefficient;  // For flow past obstacle
    float liftCoefficient;
    float strouhalNumber;
};

class BenchmarkCase
{
public:
    virtual ~BenchmarkCase() = default;

    virtual void initialize(cuda::Simulation& simulation, const nlohmann::json& config) = 0;
    virtual void applyBoundaryConditions(cuda::Simulation& simulation, float time) = 0;
    virtual BenchmarkResults analyze(const cuda::Simulation& simulation) = 0;
};

class BenchmarkFramework
{
public:
    BenchmarkFramework(const BenchmarkParameters& params)
        : _params(params),
          _memoryBuffer()
    {
        _testCase = createTestCase(params.testCase);

        // Initialize output directory
        std::filesystem::create_directories(params.outputPath);
    }

    void runBenchmarks()
    {
        if (!_testCase)
        {
            panda::log::Error("Invalid test case specified: {}", _params.testCase);
            return;
        }

        panda::log::Info("Starting benchmark: {}", _params.testCase);

        // Run the three simulations
        runSimulation(SimulationType::Fine);
        runSimulation(SimulationType::Coarse);
        runSimulation(SimulationType::Adaptive);

        // Compare results
        compareSimulations();

        // Save results
        saveResults(_params.outputPath + _params.testCase + "_results.json");
        saveResultsCSV(_params.outputPath + _params.testCase);

        panda::log::Info("Benchmark completed: {}", _params.testCase);
    }

private:
    enum class SimulationType
    {
        Fine,
        Coarse,
        Adaptive
    };

    std::unique_ptr<BenchmarkCase> createTestCase(const std::string& testName);

    void runSimulation(SimulationType type)
    {
        // Create simulation configuration based on type
        nlohmann::json config;
        config["reynoldsNumber"] = _params.reynoldsNumber;

        // Prepare base simulation parameters
        cuda::Simulation::Parameters simParams;

        // Configure simulation based on type
        std::string typeName;
        switch (type)
        {
        case SimulationType::Fine:
            typeName = "fine";
            config["particleSpacing"] = _params.fine.particleSize;
            simParams.baseParticleRadius = _params.fine.particleSize / 2.0f;
            simParams.baseSmoothingRadius = 2.5f * _params.fine.particleSize;
            // Disable adaptivity for fine simulation
            _refinementParams.enabled = false;
            break;

        case SimulationType::Coarse:
            typeName = "coarse";
            config["particleSpacing"] = _params.coarse.particleSize;
            simParams.baseParticleRadius = _params.coarse.particleSize / 2.0f;
            simParams.baseSmoothingRadius = 2.5f * _params.coarse.particleSize;
            // Disable adaptivity for coarse simulation
            _refinementParams.enabled = false;
            break;

        case SimulationType::Adaptive:
            typeName = "adaptive";
            config["particleSpacing"] = _params.adaptive.maxParticleSize;
            simParams.baseParticleRadius = _params.adaptive.maxParticleSize / 2.0f;
            simParams.baseSmoothingRadius = 2.5f * _params.adaptive.maxParticleSize;
            // Enable adaptivity
            _refinementParams.enabled = true;
            _refinementParams.minMassRatio =
                std::pow(_params.adaptive.minParticleSize / _params.adaptive.maxParticleSize, 3.0f);
            break;
        }

        panda::log::Info("Running {} simulation for {}", typeName, _params.testCase);

        // Create and initialize simulation
        auto simulation = createSimulation(simParams, {}, _memoryBuffer, _refinementParams);
        _testCase->initialize(*simulation, config);

        // Run simulation loop
        BenchmarkResults results;
        results.timeSteps.reserve(_params.totalSimulationFrames / _params.measurementInterval);
        results.performanceMetrics.reserve(_params.totalSimulationFrames / _params.measurementInterval);

        for (uint32_t frame = 0; frame < _params.totalSimulationFrames; frame++)
        {
            // Apply boundary conditions
            _testCase->applyBoundaryConditions(*simulation, frame * 0.01f);

            // Measure performance
            auto startTime = std::chrono::high_resolution_clock::now();

            // Run simulation step
            simulation->update(simParams, 0.01f);

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            // Record metrics at intervals
            if (frame % _params.measurementInterval == 0)
            {
                panda::log::Info("Frame {}/{}: {} particles",
                                 frame,
                                 _params.totalSimulationFrames,
                                 simulation->getParticlesCount());

                results.timeSteps.push_back(frame * 0.01f);

                // Performance metrics
                PerformanceMetrics perf;
                perf.simulationTimeMs = duration.count();
                perf.particleCount = simulation->getParticlesCount();
                perf.frameTimeMs = duration.count();                                 // Same for now
                perf.memoryUsageBytes = perf.particleCount * sizeof(glm::vec4) * 5;  // Approximation
                results.performanceMetrics.push_back(perf);

                // Conservation metrics
                ConservationMetrics cons = calculateConservation(*simulation);
                results.conservationMetrics.push_back(cons);
            }
        }

        // Analyze final state
        BenchmarkResults finalAnalysis = _testCase->analyze(*simulation);

        // Merge metrics
        finalAnalysis.timeSteps = results.timeSteps;
        finalAnalysis.performanceMetrics = results.performanceMetrics;
        finalAnalysis.conservationMetrics = results.conservationMetrics;

        // Store results based on simulation type
        switch (type)
        {
        case SimulationType::Fine:
            _fineResults = finalAnalysis;
            break;
        case SimulationType::Coarse:
            _coarseResults = finalAnalysis;
            break;
        case SimulationType::Adaptive:
            _adaptiveResults = finalAnalysis;
            break;
        }
    }

    ConservationMetrics calculateConservation(const cuda::Simulation& simulation)
    {
        ConservationMetrics metrics = {0};

        auto particles = simulation.getParticles();

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            // Mass conservation
            metrics.totalMass += particles.masses[i];

            // Kinetic energy
            const auto vel = glm::vec3(particles.velocities[i]);
            metrics.totalKineticEnergy += 0.5f * particles.masses[i] * glm::dot(vel, vel);

            // Momentum
            metrics.totalMomentum[0] += particles.masses[i] * vel.x;
            metrics.totalMomentum[1] += particles.masses[i] * vel.y;
            metrics.totalMomentum[2] += particles.masses[i] * vel.z;
        }

        return metrics;
    }

    void compareSimulations()
    {
        if (_fineResults.timeSteps.empty() || _coarseResults.timeSteps.empty() || _adaptiveResults.timeSteps.empty())
        {
            panda::log::Error("One or more simulations failed to produce results");
            return;
        }

        // Calculate errors between adaptive and fine simulations
        _adaptiveVsFineErrors = calculateErrors(_adaptiveResults, _fineResults);

        // Calculate errors between adaptive and coarse simulations
        _adaptiveVsCoarseErrors = calculateErrors(_adaptiveResults, _coarseResults);

        panda::log::Info("Comparison completed");
        panda::log::Info("Adaptive vs Fine max velocity error: {:.4f}",
                         *std::max_element(_adaptiveVsFineErrors.velocityErrors.begin(),
                                           _adaptiveVsFineErrors.velocityErrors.end()));
        panda::log::Info("Adaptive vs Coarse max velocity error: {:.4f}",
                         *std::max_element(_adaptiveVsCoarseErrors.velocityErrors.begin(),
                                           _adaptiveVsCoarseErrors.velocityErrors.end()));
    }

    struct ErrorMetrics
    {
        std::vector<float> velocityErrors;
        std::vector<float> densityErrors;
    };

    ErrorMetrics calculateErrors(const BenchmarkResults& results1, const BenchmarkResults& results2)
    {
        ErrorMetrics errors;

        // Calculate velocity profile errors
        if (!results1.vxProfile.empty() && !results2.vxProfile.empty())
        {
            errors.velocityErrors.push_back(calculateProfileError(results1.vxProfile, results2.vxProfile));
        }

        if (!results1.vyProfile.empty() && !results2.vyProfile.empty())
        {
            errors.velocityErrors.push_back(calculateProfileError(results1.vyProfile, results2.vyProfile));
        }

        // If no profiles available, use performance time series for errors
        if (errors.velocityErrors.empty() && results1.performanceMetrics.size() == results2.performanceMetrics.size())
        {
            for (size_t i = 0; i < results1.performanceMetrics.size(); i++)
            {
                // Calculate normalized error
                float particleRatio = static_cast<float>(results2.performanceMetrics[i].particleCount) /
                                      static_cast<float>(results1.performanceMetrics[i].particleCount);

                // Error inversely proportional to resolution ratio
                errors.velocityErrors.push_back(1.0f / particleRatio);
            }
        }

        return errors;
    }

    float calculateProfileError(const std::vector<glm::vec2>& profile1, const std::vector<glm::vec2>& profile2)
    {
        // Interpolate profile2 to profile1 positions
        std::vector<float> interpolatedValues;
        interpolatedValues.reserve(profile1.size());

        for (const auto& point1 : profile1)
        {
            // Find bracketing points in profile2
            size_t lowerIdx = 0;
            while (lowerIdx < profile2.size() - 1 && profile2[lowerIdx].x < point1.x)
            {
                lowerIdx++;
            }

            if (lowerIdx == 0 || lowerIdx == profile2.size() - 1)
            {
                // Use nearest value at boundaries
                float value = (lowerIdx == 0) ? profile2[0].y : profile2.back().y;
                interpolatedValues.push_back(value);
            }
            else
            {
                // Linear interpolation
                float x0 = profile2[lowerIdx - 1].x;
                float x1 = profile2[lowerIdx].x;
                float y0 = profile2[lowerIdx - 1].y;
                float y1 = profile2[lowerIdx].y;

                float t = (point1.x - x0) / (x1 - x0);
                float interpolatedValue = y0 + t * (y1 - y0);
                interpolatedValues.push_back(interpolatedValue);
            }
        }

        // Calculate RMS error
        float sumSquaredDiff = 0.0f;
        float sumSquaredRef = 0.0f;

        for (size_t i = 0; i < profile1.size(); i++)
        {
            float diff = profile1[i].y - interpolatedValues[i];
            sumSquaredDiff += diff * diff;
            sumSquaredRef += profile1[i].y * profile1[i].y;
        }

        return std::sqrt(sumSquaredDiff) / (std::sqrt(sumSquaredRef) + 1e-6f);
    }

    void saveResults(const std::string& filename)
    {
        nlohmann::json j;

        // Store configuration
        j["config"] = {
            {"testCase",           _params.testCase                },
            {"reynoldsNumber",     _params.reynoldsNumber          },
            {"coarseParticleSize", _params.coarse.particleSize     },
            {"fineParticleSize",   _params.fine.particleSize       },
            {"adaptiveMinSize",    _params.adaptive.minParticleSize},
            {"adaptiveMaxSize",    _params.adaptive.maxParticleSize}
        };

        // Store simulation results
        j["simulations"]["fine"] = serializeResults(_fineResults);
        j["simulations"]["coarse"] = serializeResults(_coarseResults);
        j["simulations"]["adaptive"] = serializeResults(_adaptiveResults);

        // Store error metrics
        j["errors"]["adaptiveVsFine"] = {
            {"velocityErrors",    _adaptiveVsFineErrors.velocityErrors               },
            {"meanVelocityError", calculateMean(_adaptiveVsFineErrors.velocityErrors)}
        };

        j["errors"]["adaptiveVsCoarse"] = {
            {"velocityErrors",    _adaptiveVsCoarseErrors.velocityErrors               },
            {"meanVelocityError", calculateMean(_adaptiveVsCoarseErrors.velocityErrors)}
        };

        // Write to file
        std::ofstream file(filename);
        file << j.dump(4);

        panda::log::Info("Results saved to: {}", filename);
    }

    nlohmann::json serializeResults(const BenchmarkResults& results)
    {
        nlohmann::json j;

        // Performance metrics
        j["performance"] = {
            {"maxParticleCount",
             getMaxValue(results.performanceMetrics,
             [](const PerformanceMetrics& p) {
                             return p.particleCount;
                         })                                                                 },
            {"meanSimulationTimeMs", getMeanValue(results.performanceMetrics, [](const PerformanceMetrics& p) {
                 return p.simulationTimeMs;
             })}
        };

        // Conservation metrics
        j["conservation"] = {
            {"finalMass",          results.conservationMetrics.back().totalMass         },
            {"finalKineticEnergy", results.conservationMetrics.back().totalKineticEnergy},
            {"finalMomentumX",     results.conservationMetrics.back().totalMomentum[0]  },
            {"finalMomentumY",     results.conservationMetrics.back().totalMomentum[1]  },
            {"finalMomentumZ",     results.conservationMetrics.back().totalMomentum[2]  }
        };

        // Test case specific metrics
        if (!results.vxProfile.empty() && !results.vyProfile.empty())
        {
            j["lidDrivenCavity"] = {
                {"vortexCenterX", results.vortexCenterX},
                {"vortexCenterY", results.vortexCenterY},
                {"kineticEnergy", results.kineticEnergy}
            };
        }

        return j;
    }

    template <typename T, typename F>
    float getMeanValue(const std::vector<T>& data, F selector)
    {
        if (data.empty())
        {
            return 0.0f;
        }

        float sum = 0.0f;
        for (const auto& item : data)
        {
            sum += static_cast<float>(selector(item));
        }
        return sum / static_cast<float>(data.size());
    }

    template <typename T, typename F>
    auto getMaxValue(const std::vector<T>& data, F selector)
    {
        if (data.empty())
        {
            return 0.0f;
        }

        auto maxIt = std::max_element(data.begin(), data.end(), [&selector](const T& a, const T& b) {
            return selector(a) < selector(b);
        });
        return selector(*maxIt);
    }

    float calculateMean(const std::vector<float>& values)
    {
        if (values.empty())
        {
            return 0.0f;
        }

        float sum = 0.0f;
        for (float value : values)
        {
            sum += value;
        }
        return sum / static_cast<float>(values.size());
    }

    void saveResultsCSV(const std::string& filenamePrefix)
    {
        // Save time series data for plotting

        // Velocity errors
        std::ofstream velocityErrorFile(filenamePrefix + "_velocity_errors.csv");
        velocityErrorFile << "Timestep,AdaptiveVsFine,AdaptiveVsCoarse\n";

        for (size_t i = 0; i < _fineResults.timeSteps.size() && i < _adaptiveVsFineErrors.velocityErrors.size(); i++)
        {
            velocityErrorFile << _fineResults.timeSteps[i] << ",";
            velocityErrorFile << _adaptiveVsFineErrors.velocityErrors[i] << ",";
            velocityErrorFile << _adaptiveVsCoarseErrors.velocityErrors[i] << "\n";
        }

        // Performance metrics
        std::ofstream performanceFile(filenamePrefix + "_performance.csv");
        performanceFile << "Timestep,Fine_ParticleCount,Coarse_ParticleCount,Adaptive_ParticleCount,";
        performanceFile << "Fine_SimTimeMs,Coarse_SimTimeMs,Adaptive_SimTimeMs\n";

        for (size_t i = 0; i < _fineResults.timeSteps.size(); i++)
        {
            performanceFile << _fineResults.timeSteps[i] << ",";

            if (i < _fineResults.performanceMetrics.size())
            {
                performanceFile << _fineResults.performanceMetrics[i].particleCount << ",";
            }
            else
            {
                performanceFile << "0,";
            }

            if (i < _coarseResults.performanceMetrics.size())
            {
                performanceFile << _coarseResults.performanceMetrics[i].particleCount << ",";
            }
            else
            {
                performanceFile << "0,";
            }

            if (i < _adaptiveResults.performanceMetrics.size())
            {
                performanceFile << _adaptiveResults.performanceMetrics[i].particleCount << ",";
            }
            else
            {
                performanceFile << "0,";
            }

            if (i < _fineResults.performanceMetrics.size())
            {
                performanceFile << _fineResults.performanceMetrics[i].simulationTimeMs << ",";
            }
            else
            {
                performanceFile << "0,";
            }

            if (i < _coarseResults.performanceMetrics.size())
            {
                performanceFile << _coarseResults.performanceMetrics[i].simulationTimeMs << ",";
            }
            else
            {
                performanceFile << "0,";
            }

            if (i < _adaptiveResults.performanceMetrics.size())
            {
                performanceFile << _adaptiveResults.performanceMetrics[i].simulationTimeMs;
            }
            else
            {
                performanceFile << "0";
            }

            performanceFile << "\n";
        }

        // Save velocity profiles for lid-driven cavity
        if (!_fineResults.vxProfile.empty())
        {
            std::ofstream vxProfileFile(filenamePrefix + "_vx_profile.csv");
            vxProfileFile << "Y,Fine,Coarse,Adaptive\n";

            // Normalize to fixed number of points
            const int numPoints = 20;
            for (int i = 0; i < numPoints; i++)
            {
                float y = static_cast<float>(i) / (numPoints - 1);
                vxProfileFile << y << ",";

                // Interpolate profiles at this y
                vxProfileFile << interpolateProfile(_fineResults.vxProfile, y) << ",";
                vxProfileFile << interpolateProfile(_coarseResults.vxProfile, y) << ",";
                vxProfileFile << interpolateProfile(_adaptiveResults.vxProfile, y) << "\n";
            }
        }

        if (!_fineResults.vyProfile.empty())
        {
            std::ofstream vyProfileFile(filenamePrefix + "_vy_profile.csv");
            vyProfileFile << "X,Fine,Coarse,Adaptive\n";

            // Normalize to fixed number of points
            const int numPoints = 20;
            for (int i = 0; i < numPoints; i++)
            {
                float x = static_cast<float>(i) / (numPoints - 1);
                vyProfileFile << x << ",";

                // Interpolate profiles at this x
                vyProfileFile << interpolateProfile(_fineResults.vyProfile, x) << ",";
                vyProfileFile << interpolateProfile(_coarseResults.vyProfile, x) << ",";
                vyProfileFile << interpolateProfile(_adaptiveResults.vyProfile, x) << "\n";
            }
        }

        panda::log::Info("CSV results saved to: {}_*.csv", filenamePrefix);
    }

    float interpolateProfile(const std::vector<glm::vec2>& profile, float pos)
    {
        if (profile.empty())
        {
            return 0.0f;
        }

        // Find bracketing points
        size_t lowerIdx = 0;
        while (lowerIdx < profile.size() - 1 && profile[lowerIdx].x < pos)
        {
            lowerIdx++;
        }

        if (lowerIdx == 0 || lowerIdx >= profile.size() - 1)
        {
            // Use nearest value at boundaries
            return (lowerIdx == 0) ? profile[0].y : profile.back().y;
        }

        // Linear interpolation
        float x0 = profile[lowerIdx - 1].x;
        float x1 = profile[lowerIdx].x;
        float y0 = profile[lowerIdx - 1].y;
        float y1 = profile[lowerIdx].y;

        float t = (pos - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    BenchmarkParameters _params;
    std::unique_ptr<BenchmarkCase> _testCase;
    cuda::ParticlesDataBuffer _memoryBuffer;
    cuda::refinement::RefinementParameters _refinementParams;

    BenchmarkResults _fineResults;
    BenchmarkResults _coarseResults;
    BenchmarkResults _adaptiveResults;

    ErrorMetrics _adaptiveVsFineErrors;
    ErrorMetrics _adaptiveVsCoarseErrors;
};

}  // namespace sph
*/
