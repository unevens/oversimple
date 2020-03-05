/*
Copyright 2020 Dario Mambro

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
#include "oversimple/FirOversampling.hpp"
#include "oversimple/IirOversamplingFactory.hpp"
#include <functional>

namespace oversimple {

/**
 * A class to control an Oversampling instance. It contains simple fields to
 * setup how to oversample and if any upsampled audio buffers are needed.
 * Then the Oversampling instance will do all the work.
 * @see Oversampling
 */

struct OversamplingSettings
{
  int numChannels;

  int numScalarToVecUpsamplers;
  int numVecToVecUpsamplers;
  int numScalarToScalarUpsamplers;
  int numScalarToScalarDownsamplers;
  int numVecToScalarDownsamplers;
  int numVecToVecDownsamplers;

  int numScalarBuffers;
  int numInterleavedBuffers;

  int order;
  bool linearPhase;
  int numSamplesPerBlock;
  double firTransitionBand;

  std::function<void(int)> updateLatency;

  OversamplingSettings(std::function<void(int)> updateLatency = nullptr,
                       int numChannels = 2,
                       int numScalarToVecUpsamplers = 0,
                       int numVecToScalarDownsamplers = 0,
                       int numScalarToScalarUpsamplers = 0,
                       int numScalarToScalarDownsamplers = 0,
                       int numVecToVecUpsamplers = 0,
                       int numVecToVecDownsamplers = 0,
                       int numScalarBuffers = 0,
                       int numInterleavedBuffers = 0,
                       double firTransitionBand = 4.0,
                       int order = 0,
                       bool linearPhase = false,
                       int numSamplesPerBlock = 256)
    : updateLatency(updateLatency)
    , numChannels(numChannels)
    , numScalarToVecUpsamplers(numScalarToVecUpsamplers)
    , numVecToScalarDownsamplers(numVecToScalarDownsamplers)
    , numScalarToScalarDownsamplers(numScalarToScalarDownsamplers)
    , numScalarToScalarUpsamplers(numScalarToScalarUpsamplers)
    , numVecToVecUpsamplers(numVecToVecUpsamplers)
    , numVecToVecDownsamplers(numVecToVecDownsamplers)
    , order(order)
    , linearPhase(linearPhase)
    , numSamplesPerBlock(numSamplesPerBlock)
    , firTransitionBand(firTransitionBand)
    , numScalarBuffers(numScalarBuffers)
    , numInterleavedBuffers(numInterleavedBuffers)
  {}
};

/**
 * A class to abstract over all the implementations in this library, which you
 * can control with an OversamplingSettings object, and offers a simple api for
 * oversampling and management of upsampled audio buffers.
 * @see OversamplingSettings
 */
template<typename Scalar>
class Oversampling
{
  int numSamplesPerBlock;
  int rate;

public:
  class VecToVecUpsampler
  {
    std::unique_ptr<IirUpsampler<Scalar>> iirUpsampler;
    std::unique_ptr<TFirUpsampler<Scalar>> firUpsampler;
    ScalarBuffer<Scalar> firInputBuffer;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    InterleavedBuffer<Scalar>& getOutput() { return outputBuffer; }

    int processBlock(InterleavedBuffer<Scalar> const& input,
                     int numChannelsToUpsample,
                     int numSamples)
    {
      if (firUpsampler) {
        bool ok = input.deinterleave(
          firInputBuffer.get(), numChannelsToUpsample, numSamples);
        assert(ok);
        int const numUpsampledSamples =
          firUpsampler->processBlock(firInputBuffer.get(),
                                     numChannelsToUpsample,
                                     numSamples,
                                     firOutputBuffer);
        ok = outputBuffer.interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->processBlock(
          input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpsampler->getOrder());
      }
    }

    void prepareBuffers(int numSamples)
    {
      int const oversamplingRate = firUpsampler
                                     ? firUpsampler->getRate()
                                     : (1 << iirUpsampler->getOrder());
      int const numUpsampledSamples = numSamples * oversamplingRate;

      outputBuffer.setNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->prepareBuffers(numSamples);
        firInputBuffer.setNumSamples(numSamples);
        firOutputBuffer.setNumSamples(numUpsampledSamples);
      }
      if (iirUpsampler) {
        iirUpsampler->prepareBuffer(numSamples);
      }
    }

    VecToVecUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firUpsampler->setRate(1 << settings.order);
        iirUpsampler = nullptr;
        firOutputBuffer.setNumChannels(settings.numChannels);
        firInputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<double>::make(settings.numChannels);
        iirUpsampler->setOrder(settings.order);
        firOutputBuffer.setNumChannels(0);
        firInputBuffer.setNumChannels(0);
        firUpsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock);
    }

    int getLatency()
    {
      if (firUpsampler) {
        return firUpsampler->getNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int getMaxUpsampledSamples()
    {
      if (firUpsampler) {
        return firUpsampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    void reset()
    {
      if (firUpsampler) {
        firUpsampler->reset();
      }
      if (iirUpsampler) {
        iirUpsampler->reset();
      }
    }

    int getRate() const
    {
      if (firUpsampler) {
        return firUpsampler->getRate();
      }
      if (iirUpsampler) {
        return 1 << iirUpsampler->getOrder();
      }
    }
  };

  class ScalarToVecUpsampler
  {
    std::unique_ptr<IirUpsampler<Scalar>> iirUpsampler;
    std::unique_ptr<TFirUpsampler<Scalar>> firUpsampler;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    InterleavedBuffer<Scalar>& getOutput() { return outputBuffer; }

    int processBlock(Scalar* const* input,
                     int numChannelsToUpsample,
                     int numSamples)
    {
      if (firUpsampler) {
        int numUpsampledSamples = firUpsampler->processBlock(
          input, numChannelsToUpsample, numSamples, firOutputBuffer);
        bool ok = outputBuffer.interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->processBlock(
          input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpsampler->getOrder());
      }
    }

    void prepareBuffers(int numSamples)
    {
      int const oversamplingRate = firUpsampler
                                     ? firUpsampler->getRate()
                                     : (1 << iirUpsampler->getOrder());
      int const numUpsampledSamples = numSamples * oversamplingRate;
      outputBuffer.setNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->prepareBuffers(numSamples);
        firOutputBuffer.setNumSamples(numUpsampledSamples);
      }
      if (iirUpsampler) {
        iirUpsampler->prepareBuffer(numSamples);
      }
    }

    ScalarToVecUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firUpsampler->setRate(1 << settings.order);
        iirUpsampler = nullptr;
        firOutputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<double>::make(settings.numChannels);
        iirUpsampler->setOrder(settings.order);
        firOutputBuffer.setNumChannels(0);
        firUpsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock);
    }

    int getLatency()
    {
      if (firUpsampler) {
        return firUpsampler->getNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int getMaxUpsampledSamples()
    {
      if (firUpsampler) {
        return firUpsampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    void reset()
    {
      if (firUpsampler) {
        firUpsampler->reset();
      }
      if (iirUpsampler) {
        iirUpsampler->reset();
      }
    }

    int getRate() const
    {
      if (firUpsampler) {
        return firUpsampler->getRate();
      }
      if (iirUpsampler) {
        return 1 << iirUpsampler->getOrder();
      }
    }
  };

  class ScalarToScalarUpsampler
  {
    std::unique_ptr<IirUpsampler<Scalar>> iirUpsampler;
    std::unique_ptr<TFirUpsampler<Scalar>> firUpsampler;
    ScalarBuffer<Scalar> outputBuffer;
    InterleavedBuffer<Scalar> iirOutputBuffer;

  public:
    ScalarBuffer<Scalar>& getOutput() { return outputBuffer; }

    int processBlock(Scalar* const* input,
                     int numChannelsToUpsample,
                     int numSamples)
    {
      if (firUpsampler) {
        int numUpsampledSamples = firUpsampler->processBlock(
          input, numChannelsToUpsample, numSamples, outputBuffer);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->processBlock(
          input, numSamples, iirOutputBuffer, numChannelsToUpsample);
        iirOutputBuffer.deinterleave(outputBuffer.get(),
                                     numChannelsToUpsample,
                                     iirOutputBuffer.getNumSamples());
        return numSamples * (1 << iirUpsampler->getOrder());
      }
    }

    void prepareBuffers(int numSamples)
    {
      int const oversamplingRate = firUpsampler
                                     ? firUpsampler->getRate()
                                     : (1 << iirUpsampler->getOrder());
      int const numUpsampledSamples = numSamples * oversamplingRate;
      outputBuffer.setNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->prepareBuffers(numSamples);
      }
      if (iirUpsampler) {
        iirOutputBuffer.setNumSamples(numUpsampledSamples);
        iirUpsampler->prepareBuffer(numSamples);
      }
    }

    ScalarToScalarUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firUpsampler->setRate(1 << settings.order);
        iirUpsampler = nullptr;
        iirOutputBuffer.setNumChannels(0);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<double>::make(settings.numChannels);
        iirUpsampler->setOrder(settings.order);
        iirOutputBuffer.setNumChannels(settings.numChannels);
        firUpsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock);
    }

    int getLatency()
    {
      if (firUpsampler) {
        return firUpsampler->getNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int getMaxUpsampledSamples()
    {
      if (firUpsampler) {
        return firUpsampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    void reset()
    {
      if (firUpsampler) {
        firUpsampler->reset();
      }
      if (iirUpsampler) {
        iirUpsampler->reset();
      }
    }

    int getRate() const
    {
      if (firUpsampler) {
        return firUpsampler->getRate();
      }
      if (iirUpsampler) {
        return 1 << iirUpsampler->getOrder();
      }
    }
  };

  struct VecToScalarDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    ScalarBuffer<Scalar> firInputBuffer;

  public:
    VecToScalarDownsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firDownsampler = std::make_unique<TFirDownsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firDownsampler->setRate(1 << settings.order);
        iirDownsampler = nullptr;
        firInputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirDownsampler =
          IirDownsamplerFactory<double>::make(settings.numChannels);
        iirDownsampler->setOrder(settings.order);
        firInputBuffer.setNumChannels(0);
        firDownsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock,
                     settings.numSamplesPerBlock * (1 << settings.order));
    }

    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownsampler
                                     ? firDownsampler->getRate()
                                     : (1 << iirDownsampler->getOrder());
      if (firDownsampler) {
        firDownsampler->prepareBuffers(maxNumUpsampledSamples, numSamples);
        firInputBuffer.setNumSamples(maxNumUpsampledSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->prepareBuffer(numSamples);
      }
    }

    void processBlock(InterleavedBuffer<Scalar> const& input,
                      Scalar** output,
                      int numChannelsToDownsample,
                      int numSamples)
    {
      if (firDownsampler) {
        input.deinterleave(firInputBuffer);
        firDownsampler->processBlock(
          firInputBuffer, output, numChannelsToDownsample, numSamples);
      }
      else {
        iirDownsampler->processBlock(
          input, numSamples * (1 << iirDownsampler->getOrder()));
        iirDownsampler->getOutput().deinterleave(
          output, numChannelsToDownsample, numSamples);
      }
    }

    int getLatency()
    {
      if (firDownsampler) {
        return (double)firDownsampler->getNumSamplesBeforeOutputStarts() /
               (double)firDownsampler->getRate();
      }
      return 0;
    }

    void reset()
    {
      if (firDownsampler) {
        firDownsampler->reset();
      }
      if (iirDownsampler) {
        iirDownsampler->reset();
      }
    }

    int getRate() const
    {
      if (firDownsampler) {
        return firDownsampler->getRate();
      }
      if (iirDownsampler) {
        return 1 << iirDownsampler->getOrder();
      }
    }
  };

  struct VecToVecDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    ScalarBuffer<Scalar> firInputBuffer;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    VecToVecDownsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firDownsampler = std::make_unique<TFirDownsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firDownsampler->setRate(1 << settings.order);
        iirDownsampler = nullptr;
        firInputBuffer.setNumChannels(settings.numChannels);
        firOutputBuffer.setNumChannels(settings.numChannels);
        outputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirDownsampler =
          IirDownsamplerFactory<double>::make(settings.numChannels);
        iirDownsampler->setOrder(settings.order);
        firInputBuffer.setNumChannels(0);
        firOutputBuffer.setNumChannels(0);
        outputBuffer.setNumChannels(0);
        firDownsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock,
                     settings.numSamplesPerBlock * (1 << settings.order));
    }

    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownsampler
                                     ? firDownsampler->getRate()
                                     : (1 << iirDownsampler->getOrder());
      if (firDownsampler) {
        firDownsampler->prepareBuffers(maxNumUpsampledSamples, numSamples);
        firInputBuffer.setNumSamples(maxNumUpsampledSamples);
        firOutputBuffer.setNumSamples(numSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->prepareBuffer(numSamples);
      }
      outputBuffer.setNumSamples(numSamples);
    }

    void processBlock(InterleavedBuffer<Scalar> const& input,
                      int numChannelsToDownsample,
                      int numSamples)
    {
      if (firDownsampler) {
        input.deinterleave(firInputBuffer);
        firDownsampler->processBlock(
          firInputBuffer, firOutputBuffer, numChannelsToDownsample);
        outputBuffer.interleave(firOutputBuffer, numChannelsToDownsample);
      }
      else {
        iirDownsampler->processBlock(
          input, numSamples * (1 << iirDownsampler->getOrder()));
      }
    }

    InterleavedBuffer<Scalar>& getOutput()
    {
      return firDownsampler ? outputBuffer : iirDownsampler->getOutput();
    }

    int getLatency()
    {
      if (firDownsampler) {
        return (double)firDownsampler->getNumSamplesBeforeOutputStarts() /
               (double)firDownsampler->getRate();
      }
      return 0;
    }

    void reset()
    {
      if (firDownsampler) {
        firDownsampler->reset();
      }
      if (iirDownsampler) {
        iirDownsampler->reset();
      }
    }

    int getRate() const
    {
      if (firDownsampler) {
        return firDownsampler->getRate();
      }
      if (iirDownsampler) {
        return 1 << iirDownsampler->getOrder();
      }
    }
  };

  struct ScalarToScalarDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    InterleavedBuffer<Scalar> iirInputBuffer;

  public:
    ScalarToScalarDownsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firDownsampler = std::make_unique<TFirDownsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firDownsampler->setRate(1 << settings.order);
        iirDownsampler = nullptr;
        iirInputBuffer.setNumChannels(0);
      }
      else {
        iirDownsampler =
          IirDownsamplerFactory<double>::make(settings.numChannels);
        iirDownsampler->setOrder(settings.order);
        iirInputBuffer.setNumChannels(2);
        firDownsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock,
                     settings.numSamplesPerBlock * (1 << settings.order));
    }

    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownsampler
                                     ? firDownsampler->getRate()
                                     : (1 << iirDownsampler->getOrder());
      if (firDownsampler) {
        firDownsampler->prepareBuffers(maxNumUpsampledSamples, numSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->prepareBuffer(numSamples);
        iirInputBuffer.setNumSamples(maxNumUpsampledSamples);
      }
    }

    void processBlock(Scalar* const* input,
                      Scalar** output,
                      int numChannelsToDownsample,
                      int numSamples)
    {
      if (firDownsampler) {
        firDownsampler->processBlock(input,
                                     numChannelsToDownsample,
                                     output,
                                     numChannelsToDownsample,
                                     numSamples);
      }
      else {
        iirInputBuffer.interleave(input, numChannelsToDownsample, numSamples);
        iirDownsampler->processBlock(
          iirInputBuffer, numSamples * (1 << iirDownsampler->getOrder()));
        iirDownsampler->getOutput().deinterleave(
          output, numChannelsToDownsample, numSamples);
      }
    }

    int getLatency()
    {
      if (firDownsampler) {
        return (double)firDownsampler->getNumSamplesBeforeOutputStarts() /
               (double)firDownsampler->getRate();
      }
      return 0;
    }

    void reset()
    {
      if (firDownsampler) {
        firDownsampler->reset();
      }
      if (iirDownsampler) {
        iirDownsampler->reset();
      }
    }

    int getRate() const
    {
      if (firDownsampler) {
        return firDownsampler->getRate();
      }
      if (iirDownsampler) {
        return 1 << iirDownsampler->getOrder();
      }
    }
  };

  std::vector<std::unique_ptr<ScalarToVecUpsampler>> scalarToVecUpsamplers;

  std::vector<std::unique_ptr<VecToVecUpsampler>> vecToVecUpsamplers;

  std::vector<std::unique_ptr<ScalarToScalarUpsampler>>
    scalarToScalarUpsamplers;

  std::vector<std::unique_ptr<VecToScalarDownsampler>> vecToScalarDownsamplers;

  std::vector<std::unique_ptr<VecToVecDownsampler>> vecToVecDownsamplers;

  std::vector<std::unique_ptr<ScalarToScalarDownsampler>>
    scalarToScalarDownsamplers;

  std::vector<InterleavedBuffer<Scalar>> interleavedBuffers;

  std::vector<ScalarBuffer<Scalar>> scalarBuffers;

  Oversampling(OversamplingSettings const& settings)
    : numSamplesPerBlock(settings.numSamplesPerBlock)
    , rate(1 << settings.order)
  {
    for (int i = 0; i < settings.numScalarToVecUpsamplers; ++i) {
      scalarToVecUpsamplers.push_back(
        std::make_unique<ScalarToVecUpsampler>(settings));
    }
    for (int i = 0; i < settings.numVecToVecUpsamplers; ++i) {
      vecToVecUpsamplers.push_back(
        std::make_unique<VecToVecUpsampler>(settings));
    }
    for (int i = 0; i < settings.numScalarToScalarUpsamplers; ++i) {
      scalarToScalarUpsamplers.push_back(
        std::make_unique<ScalarToScalarUpsampler>(settings));
    }
    for (int i = 0; i < settings.numVecToScalarDownsamplers; ++i) {
      vecToScalarDownsamplers.push_back(
        std::make_unique<VecToScalarDownsampler>(settings));
    }
    for (int i = 0; i < settings.numVecToVecDownsamplers; ++i) {
      vecToVecDownsamplers.push_back(
        std::make_unique<VecToVecDownsampler>(settings));
    }
    for (int i = 0; i < settings.numScalarToScalarDownsamplers; ++i) {
      scalarToScalarDownsamplers.push_back(
        std::make_unique<ScalarToScalarDownsampler>(settings));
    }
    if (settings.updateLatency) {
      settings.updateLatency(getLatency());
    }
    int maxNumUpsampledSamples = rate * settings.numSamplesPerBlock;
    interleavedBuffers.resize(settings.numInterleavedBuffers);
    for (auto& buffer : interleavedBuffers) {
      buffer.setNumSamples(maxNumUpsampledSamples);
      buffer.setNumChannels(settings.numChannels);
    }
    scalarBuffers.resize(settings.numScalarBuffers);
    for (auto& buffer : scalarBuffers) {
      buffer.setNumSamples(maxNumUpsampledSamples);
      buffer.setNumChannels(settings.numChannels);
    }
  }

  void prepareBuffers(int numSamples)
  {
    if (numSamplesPerBlock < numSamples) {

      numSamplesPerBlock = numSamples;

      for (auto& upsampler : scalarToVecUpsamplers) {
        upsampler->prepareBuffers(numSamples);
      }
      for (auto& upsampler : vecToVecUpsamplers) {
        upsampler->prepareBuffers(numSamples);
      }
      for (auto& upsampler : scalarToScalarUpsamplers) {
        upsampler->prepareBuffers(numSamples);
      }

      int maxNumUpsampledSamples = rate * numSamples;

      if (scalarToVecUpsamplers.size() > 0) {
        maxNumUpsampledSamples =
          scalarToVecUpsamplers[0]->getMaxUpsampledSamples();
      }
      else if (vecToVecUpsamplers.size() > 0) {
        maxNumUpsampledSamples =
          vecToVecUpsamplers[0]->getMaxUpsampledSamples();
      }

      for (auto& downsampler : scalarToScalarDownsamplers) {
        downsampler->prepareBuffers(numSamples, maxNumUpsampledSamples);
      }
      for (auto& downsampler : vecToScalarDownsamplers) {
        downsampler->prepareBuffers(numSamples, maxNumUpsampledSamples);
      }
      for (auto& downsampler : vecToVecDownsamplers) {
        downsampler->prepareBuffers(numSamples, maxNumUpsampledSamples);
      }

      for (auto& buffer : interleavedBuffers) {
        buffer.setNumSamples(maxNumUpsampledSamples);
      }

      for (auto& buffer : scalarBuffers) {
        buffer.setNumSamples(maxNumUpsampledSamples);
      }
    }
  }

  int getLatency()
  {
    int latency = 0;
    if (scalarToVecUpsamplers.size() > 0) {
      latency += scalarToVecUpsamplers[0]->getLatency();
    }
    else if (vecToVecUpsamplers.size() > 0) {
      latency += vecToVecUpsamplers[0]->getLatency();
    }
    if (vecToScalarDownsamplers.size() > 0) {
      latency += vecToScalarDownsamplers[0]->getLatency();
    }
    else if (scalarToScalarDownsamplers.size() > 0) {
      latency += scalarToScalarDownsamplers[0]->getLatency();
    }
    else if (vecToVecDownsamplers.size() > 0) {
      latency += vecToVecDownsamplers[0]->getLatency();
    }
    return latency;
  }

  void reset()
  {
    for (auto& upsampler : scalarToVecUpsamplers) {
      upsampler->reset();
    }
    for (auto& upsampler : vecToVecUpsamplers) {
      upsampler->reset();
    }
    for (auto& downsampler : scalarToScalarDownsamplers) {
      downsampler->reset();
    }
    for (auto& downsampler : vecToVecDownsamplers) {
      downsampler->reset();
    }
    for (auto& downsampler : vecToScalarDownsamplers) {
      downsampler->reset();
    }
  }

  int getRate() const { return rate; }

  int getNumSamplesPerBlock() const { return numSamplesPerBlock; }
};

} // namespace oversimple
