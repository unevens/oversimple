/*
Copyright 2019*2021 Dario Mambro

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

namespace oversimple::iir::detail {

/**
 * Class to design IIR antialiasing filters.
 */

class OversamplingDesigner final
{
  struct Stage
  {
    double const attenuation;
    double const transition;
    uint32_t const numCoefs;
    double getGroupDelay(double normalizedFrequency) const;
    double getPhaseDelay(double normalizedFrequency) const;
    double getMaxGroupDelay() const;
    double getMinGroupDelay() const;
    Stage(double attenuation, double transition);
    Stage Next() const
    {
      return Stage(attenuation, 0.5 * (0.5 + transition));
    }
    std::string print() const;
    std::vector<double> computeCoefs() const;
    void computeCoefs(std::vector<double>& coefs) const;
  };

  class GroupDelayGraph
  {
    std::vector<double> graph;
    void fromStages(std::vector<Stage> const& stages, uint32_t resolution);

  public:
    GroupDelayGraph(std::vector<Stage> const& stages, uint32_t resolution);
    double getMean() const;
    std::vector<double> getGraph() const
    {
      return graph;
    }
  };

  std::vector<Stage> stages;

public:
  /**
   * Constructor.
   * @param attenuation required stopband attenuation in dB
   * @param transition required normalized transition bandwidth
   * @param numStages number of oversampling stages
   */
  OversamplingDesigner(double attenuation, double transition, uint32_t numStages = 5);

  /**
   * @return a reference to the vector with the information specific to each
   * stage.
   */
  std::vector<Stage> const& getStages() const
  {
    return stages;
  }

  /**
   * @return the filter statistics.
   */
  std::string print() const;

  /**
   * @return data to graph the group delay.
   */
  GroupDelayGraph getGroupDelayGraph(uint32_t resolution) const
  {
    return GroupDelayGraph(stages, resolution);
  }

  /**
   * @param normalizedFrequency
   * @param oversamplingOrder
   * @return the group delay at the specified normalized frequency and
   * oversampling order
   */
  double getGroupDelay(double normalizedFrequency, uint32_t oversamplingOrder) const;

  /**
   * @param normalizedFrequency
   * @param oversamplingOrder
   * @return the phase delay at the specified normalized frequency and
   * oversampling order
   */
  double getPhaseDelay(double normalizedFrequency, uint32_t oversamplingOrder) const;

  /**
   * @param oversamplingOrder
   * @return the minimum group delay at the specified oversampling order
   */
  double getMinGroupDelay(uint32_t oversamplingOrder) const;
};

// implementation

inline OversamplingDesigner::OversamplingDesigner(double attenuation, double transition, uint32_t numStages)
{
  assert(numStages > 0);
  stages.push_back(Stage(attenuation, transition));
  for (uint32_t i = 1; i < numStages; ++i) {
    stages.push_back(stages.back().Next());
  }
}

inline std::string OversamplingDesigner::print() const
{
  std::string text;
  uint32_t i = 0;
  for (auto& stage : stages) {
    text += "stage " + std::to_string(i++) + ": " + stage.print() + "\n";
  }
  i = 0;
  double coef = 0.5;
  double min = 0.0;
  double max = 0.0;
  for (auto& stage : stages) {
    text += "group delay at order " + std::to_string(i + 1) + ": ";
    double stageMin = coef * stage.getMinGroupDelay();
    double stageMax = coef * stage.getMaxGroupDelay();
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
  text += "linear badwith at 44.1 KHz = " + std::to_string(22050.0 - baseTransition * 44100.0) + " Hz\n";
  text += "linear badwith at 96 KHz = " + std::to_string(48000.0 - baseTransition * 96000.0) + " Hz\n";
  text += "linear badwith at 192 KHz = " + std::to_string(96000.0 - baseTransition * 192000.0) + " Hz\n";
  text += "\n";
  return text;
}

inline double OversamplingDesigner::getMinGroupDelay(uint32_t order) const
{
  assert(order <= stages.size());
  double coef = 0.5;
  double latency = 0.0;
  for (uint32_t i = 0; i < order; ++i) {
    latency += coef * stages[i].getMinGroupDelay();
    coef *= 0.5;
  }
  return latency;
}

inline double OversamplingDesigner::getGroupDelay(double normalizedFrequency, uint32_t order) const
{
  assert(order <= stages.size());
  double coef = 0.5;
  double groupDelay = 0.0;
  for (uint32_t i = 0; i < order; ++i) {
    groupDelay += coef * stages[i].getGroupDelay(normalizedFrequency);
    coef *= 0.5;
  }
  return groupDelay;
}

inline double OversamplingDesigner::getPhaseDelay(double normalizedFrequency, uint32_t order) const
{
  assert(order <= stages.size());
  double coef = 0.5;
  double phaseDelay = 0.0;
  for (uint32_t i = 0; i < order; ++i) {
    phaseDelay += coef * stages[i].getPhaseDelay(normalizedFrequency);
    coef *= 0.5;
  }
  return phaseDelay;
}

inline double OversamplingDesigner::Stage::getGroupDelay(double normalizedFrequency) const
{
  std::vector<double> coefs;
  computeCoefs(coefs);
  return hiir::PolyphaseIir2Designer::compute_group_delay(&coefs[0], numCoefs, normalizedFrequency, false);
}

inline double OversamplingDesigner::Stage::getPhaseDelay(double normalizedFrequency) const
{
  std::vector<double> coefs;
  computeCoefs(coefs);
  double delay = 0.0;
  for (uint32_t i = 0; i < coefs.size(); ++i) {
    delay += hiir::PolyphaseIir2Designer::compute_phase_delay(coefs[i], normalizedFrequency);
  }
  return delay;
}

inline double OversamplingDesigner::Stage::getMinGroupDelay() const
{
  return getGroupDelay(0.0);
}

inline double OversamplingDesigner::Stage::getMaxGroupDelay() const
{
  return getGroupDelay(0.25);
}

inline OversamplingDesigner::Stage::Stage(double attenuation, double transition)
  : attenuation(attenuation)
  , transition(transition)
  , numCoefs(hiir::PolyphaseIir2Designer::compute_nbr_coefs_from_proto(attenuation, transition))
{}

inline std::string OversamplingDesigner::Stage::print() const
{
  return "transition = " + std::to_string(transition) + ", " + "numCoefs = " + std::to_string(numCoefs) + ", " +
         "attenuation = " + std::to_string(attenuation);
}

inline std::vector<double> OversamplingDesigner::Stage::computeCoefs() const
{
  std::vector<double> coefs;
  computeCoefs(coefs);
  return coefs;
}

inline void OversamplingDesigner::Stage::computeCoefs(std::vector<double>& coefs) const
{
  coefs.resize(numCoefs);
  hiir::PolyphaseIir2Designer::compute_coefs(&coefs[0], attenuation, transition);
}

inline void OversamplingDesigner::GroupDelayGraph::fromStages(std::vector<Stage> const& stages, uint32_t resolution)
{
  std::vector<double> coefs;
  graph.resize(resolution, 0.0);
  double f = 0.5 / resolution;
  double coef = 0.5;
  for (auto& stage : stages) {
    for (uint32_t i = 0; i < resolution; ++i) {
      graph[i] += coef * stage.getGroupDelay(i * f);
    }
    coef *= 0.5;
  }
}

inline OversamplingDesigner::GroupDelayGraph::GroupDelayGraph(std::vector<Stage> const& stages, uint32_t resolution)
{
  fromStages(stages, resolution);
}

inline double OversamplingDesigner::GroupDelayGraph::getMean() const
{
  return std::accumulate(graph.begin(), graph.end(), 0.0) / (double)graph.size();
}

} // namespace oversimple
