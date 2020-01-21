/*
Copyright 2019 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "hiir/PolyphaseIir2Designer.h"
#include <cassert>
#include <numeric>
#include <string>
#include <vector>

namespace oversimple {

/**
 * Class to design IIR antialiasing filters.
 */

class IirOversamplingDesigner final
{
  struct Stage
  {
    double const attenuation;
    double const transition;
    int const numCoefs;
    double GetGroupDelay(double normalizedFrequency) const;
    double GetPhaseDelay(double normalizedFrequency) const;
    double GetMaxGroupDelay() const;
    double GetMinGroupDelay() const;
    Stage(double attenuation, double transition);
    Stage Next() const { return Stage(attenuation, 0.5 * (0.5 + transition)); }
    std::string Print() const;
    std::vector<double> ComputeCoefs() const;
    void ComputeCoefs(std::vector<double>& coefs) const;
  };

  class GroupDelayGraph
  {
    std::vector<double> graph;
    void FromStages(std::vector<Stage> const& stages, int resolution);

  public:
    GroupDelayGraph(std::vector<Stage> const& stages, int resolution);
    double GetMean() const;
    std::vector<double> GetGraph() const { return graph; }
  };

  std::vector<Stage> stages;

public:
  /**
   * Constructor.
   * @param attenuation required stopband attenuation in dB
   * @param transition required normalized transition bandwidth
   * @param numStages number of oversampling stages
   */
  IirOversamplingDesigner(double attenuation,
                          double transition,
                          int numStages = 5);

  /**
   * @return a reference to the vector with the information specific to each
   * stage.
   */
  std::vector<Stage> const& GetStages() const { return stages; }

  /**
   * @return the filter statistics.
   */
  std::string Print() const;

  /**
   * @return data to graph the group delay.
   */
  GroupDelayGraph GetGroupDelayGraph(int resolution) const
  {
    return GroupDelayGraph(stages, resolution);
  }

  /**
   * @param normalizedFrequency
   * @param oversamplingOrder
   * @return the group delay at the specified normalized frequency and
   * oversampling order
   */
  double GetGroupDelay(double normalizedFrequency, int oversamplingOrder) const;

  /**
   * @param normalizedFrequency
   * @param oversamplingOrder
   * @return the phase delay at the specified normalized frequency and
   * oversampling order
   */
  double GetPhaseDelay(double normalizedFrequency, int oversamplingOrder) const;

  /**
   * @param oversamplingOrder
   * @return the minimum group delay at the specified oversampling order
   */
  double GetMinGroupDelay(int oversamplingOrder) const;
};

// implementation

inline IirOversamplingDesigner::IirOversamplingDesigner(double attenuation,
                                                        double transition,
                                                        int numStages)
{
  assert(numStages > 0);
  stages.push_back(Stage(attenuation, transition));
  for (int i = 1; i < numStages; ++i) {
    stages.push_back(stages.back().Next());
  }
}

inline std::string
IirOversamplingDesigner::Print() const
{
  std::string text;
  int i = 0;
  for (auto& stage : stages) {
    text += "stage " + std::to_string(i++) + ": " + stage.Print() + "\n";
  }
  i = 0;
  double coef = 0.5;
  double min = 0.0;
  double max = 0.0;
  for (auto& stage : stages) {
    text += "group delay at order " + std::to_string(i + 1) + ": ";
    double stageMin = coef * stage.GetMinGroupDelay();
    double stageMax = coef * stage.GetMaxGroupDelay();
    min += stageMin;
    max += stageMax;
    text += "min = " + std::to_string(min) + ", ";
    text += "max = " + std::to_string(max) + ". ";
    text += "(stage " + std::to_string(i++) + " group delay: ";
    text += "min = " + std::to_string(stageMin) + ", ";
    text += "max = " + std::to_string(stageMax);
    text += ")\n";
    coef *= 0.5;
  }
  double baseTransition = stages[0].transition;
  text += "linear badwith at 44.1 KHz = " +
          std::to_string(22050.0 - baseTransition * 44100.0) + " Hz\n";
  text += "linear badwith at 96 KHz = " +
          std::to_string(48000.0 - baseTransition * 96000.0) + " Hz\n";
  text += "linear badwith at 192 KHz = " +
          std::to_string(96000.0 - baseTransition * 192000.0) + " Hz\n";
  text += "\n";
  return text;
}

inline double
IirOversamplingDesigner::GetMinGroupDelay(int order) const
{
  assert(order <= stages.size());
  double coef = 0.5;
  double latency = 0.0;
  for (int i = 0; i < order; ++i) {
    latency += coef * stages[i].GetMinGroupDelay();
    coef *= 0.5;
  }
  return latency;
}

inline double
IirOversamplingDesigner::GetGroupDelay(double normalizedFrequency,
                                       int order) const
{
  assert(order <= stages.size());
  double coef = 0.5;
  double groupDelay = 0.0;
  for (int i = 0; i < order; ++i) {
    groupDelay += coef * stages[i].GetGroupDelay(normalizedFrequency);
    coef *= 0.5;
  }
  return groupDelay;
}

inline double
IirOversamplingDesigner::GetPhaseDelay(double normalizedFrequency,
                                       int order) const
{
  assert(order <= stages.size());
  double coef = 0.5;
  double phaseDelay = 0.0;
  for (int i = 0; i < order; ++i) {
    phaseDelay += coef * stages[i].GetPhaseDelay(normalizedFrequency);
    coef *= 0.5;
  }
  return phaseDelay;
}

inline double
IirOversamplingDesigner::Stage::GetGroupDelay(double normalizedFrequency) const
{
  std::vector<double> coefs;
  ComputeCoefs(coefs);
  return hiir::PolyphaseIir2Designer::compute_group_delay(
    &coefs[0], numCoefs, normalizedFrequency, false);
}

inline double
IirOversamplingDesigner::Stage::GetPhaseDelay(double normalizedFrequency) const
{
  std::vector<double> coefs;
  ComputeCoefs(coefs);
  double delay = 0.0;
  for (int i = 0; i < coefs.size(); ++i) {
    delay += hiir::PolyphaseIir2Designer::compute_phase_delay(
      coefs[i], normalizedFrequency);
  }
  return delay;
}

inline double
IirOversamplingDesigner::Stage::GetMinGroupDelay() const
{
  return GetGroupDelay(0.0);
}

inline double
IirOversamplingDesigner::Stage::GetMaxGroupDelay() const
{
  return GetGroupDelay(0.25);
}

inline IirOversamplingDesigner::Stage::Stage(double attenuation,
                                             double transition)
  : attenuation(attenuation)
  , transition(transition)
  , numCoefs(
      hiir::PolyphaseIir2Designer::compute_nbr_coefs_from_proto(attenuation,
                                                                transition))
{}

inline std::string
IirOversamplingDesigner::Stage::Print() const
{
  return "transition = " + std::to_string(transition) + ", " +
         "numCoefs = " + std::to_string(numCoefs) + ", " +
         "attenuation = " + std::to_string(attenuation);
}

inline std::vector<double>
IirOversamplingDesigner::Stage::ComputeCoefs() const
{
  std::vector<double> coefs;
  ComputeCoefs(coefs);
  return coefs;
}

inline void
IirOversamplingDesigner::Stage::ComputeCoefs(std::vector<double>& coefs) const
{
  coefs.resize(numCoefs);
  hiir::PolyphaseIir2Designer::compute_coefs(
    &coefs[0], attenuation, transition);
}

inline void
IirOversamplingDesigner::GroupDelayGraph::FromStages(
  std::vector<Stage> const& stages,
  int resolution)
{
  std::vector<double> coefs;
  graph.resize(resolution, 0.0);
  double f = 0.5 / resolution;
  double coef = 0.5;
  for (auto& stage : stages) {
    for (int i = 0; i < resolution; ++i) {
      graph[i] += coef * stage.GetGroupDelay(i * f);
    }
    coef *= 0.5;
  }
}

inline IirOversamplingDesigner::GroupDelayGraph::GroupDelayGraph(
  std::vector<Stage> const& stages,
  int resolution)
{
  FromStages(stages, resolution);
}

inline double
IirOversamplingDesigner::GroupDelayGraph::GetMean() const
{
  return std::accumulate(graph.begin(), graph.end(), 0.0) /
         (double)graph.size();
}

} // namespace oversimple
