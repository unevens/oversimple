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

  int numScalarBuffers;
  int numInterleavedBuffers;

  int order;
  bool linearPhase;
  int numSamplesPerBlock;
  double firTransitionBand;

  std::function<void(int)> UpdateLatency;

  OversamplingSettings(std::function<void(int)> UpdateLatency = nullptr,
                       int numChannels = 2,
                       int numScalarToVecUpsamplers = 0,
                       int numVecToScalarDownsamplers = 0,
                       int numScalarToScalarUpsamplers = 0,
                       int numScalarToScalarDownsamplers = 0,
                       int numVecToVecUpsamplers = 0,
                       int numScalarBuffers = 0,
                       int numInterleavedBuffers = 0,
                       double firTransitionBand = 4.0,
                       int order = 0,
                       bool linearPhase = false,
                       int numSamplesPerBlock = 256)
    : UpdateLatency(UpdateLatency)
    , numChannels(numChannels)
    , numScalarToVecUpsamplers(numScalarToVecUpsamplers)
    , numVecToScalarDownsamplers(numVecToScalarDownsamplers)
    , numScalarToScalarDownsamplers(numScalarToScalarDownsamplers)
    , numScalarToScalarUpsamplers(numScalarToScalarUpsamplers)
    , numVecToVecUpsamplers(numVecToVecUpsamplers)
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
    InterleavedBuffer<Scalar>& GetOutput() { return outputBuffer; }

    void ProcessBlock(InterleavedBuffer<Scalar>& input,
                      int numChannelsToUpsample,
                      int numSamples)
    {
      if (firUpsampler) {
        bool ok = input.Deinterleave(
          firInputBuffer.Get(), numChannelsToUpsample, numSamples);
        assert(ok);
        firUpsampler->ProcessBlock(firInputBuffer.Get(),
                                   numChannelsToUpsample,
                                   numSamples,
                                   firOutputBuffer);
        ok = outputBuffer.Interleave(firOutputBuffer, 2);
        assert(ok);
      }
      else {
        iirUpsampler->ProcessBlock(
          input, numSamples, outputBuffer, numChannelsToUpsample);
      }
    }

    void PrepareBuffers(int numSamples)
    {
      int const oversamplingFactor = firUpsampler
                                       ? firUpsampler->GetOversamplingFactor()
                                       : (1 << iirUpsampler->GetOrder());
      int const numUpsampledSamples = numSamples * oversamplingFactor;

      outputBuffer.SetNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->PrepareBuffers(numSamples);
        firInputBuffer.SetNumSamples(numSamples);
        firOutputBuffer.SetNumSamples(numUpsampledSamples);
      }
      if (iirUpsampler) {
        iirUpsampler->PrepareBuffer(numSamples);
      }
    }

    VecToVecUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firUpsampler->SetOversamplingFactor(1 << settings.order);
        iirUpsampler = nullptr;
        firOutputBuffer.SetNumChannels(settings.numChannels);
        firInputBuffer.SetNumChannels(settings.numChannels);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<double>::New(settings.numChannels);
        iirUpsampler->SetOrder(settings.order);
        firOutputBuffer.SetNumChannels(0);
        firInputBuffer.SetNumChannels(0);
        firUpsampler = nullptr;
      }
      PrepareBuffers(settings.numSamplesPerBlock);
    }

    int GetLatency()
    {
      if (firUpsampler) {
        return firUpsampler->GetNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int GetMaxUpsampledSamples()
    {
      if (firUpsampler) {
        return firUpsampler->GetMaxNumOutputSamples();
      }
      return 0;
    }

    void Reset()
    {
      if (firUpsampler) {
        firUpsampler->Reset();
      }
      if (iirUpsampler) {
        iirUpsampler->Reset();
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
    InterleavedBuffer<Scalar>& GetOutput() { return outputBuffer; }

    int ProcessBlock(Scalar** input, int numChannelsToUpsample, int numSamples)
    {
      if (firUpsampler) {
        int numUpsampledSamples = firUpsampler->ProcessBlock(
          input, numChannelsToUpsample, numSamples, firOutputBuffer);
        bool ok = outputBuffer.Interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->ProcessBlock(
          input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpsampler->GetOrder());
      }
    }

    void PrepareBuffers(int numSamples)
    {
      int const oversamplingFactor = firUpsampler
                                       ? firUpsampler->GetOversamplingFactor()
                                       : (1 << iirUpsampler->GetOrder());
      int const numUpsampledSamples = numSamples * oversamplingFactor;
      outputBuffer.SetNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->PrepareBuffers(numSamples);
        firOutputBuffer.SetNumSamples(numUpsampledSamples);
      }
      if (iirUpsampler) {
        iirUpsampler->PrepareBuffer(numSamples);
      }
    }

    ScalarToVecUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firUpsampler->SetOversamplingFactor(1 << settings.order);
        iirUpsampler = nullptr;
        firOutputBuffer.SetNumChannels(settings.numChannels);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<double>::New(settings.numChannels);
        iirUpsampler->SetOrder(settings.order);
        firOutputBuffer.SetNumChannels(0);
        firUpsampler = nullptr;
      }
      PrepareBuffers(settings.numSamplesPerBlock);
    }

    int GetLatency()
    {
      if (firUpsampler) {
        return firUpsampler->GetNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int GetMaxUpsampledSamples()
    {
      if (firUpsampler) {
        return firUpsampler->GetMaxNumOutputSamples();
      }
      return 0;
    }

    void Reset()
    {
      if (firDownsampler) {
        firDownsampler->Reset();
      }
      if (iirDownsampler) {
        iirDownsampler->Reset();
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
    ScalarBuffer<Scalar>& GetOutput() { return outputBuffer; }

    int ProcessBlock(Scalar** input, int numChannelsToUpsample, int numSamples)
    {
      if (firUpsampler) {
        int numUpsampledSamples = firUpsampler->ProcessBlock(
          input, numChannelsToUpsample, numSamples, outputBuffer);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->ProcessBlock(
          input, numSamples, iirOutputBuffer, numChannelsToUpsample);
        iirOutputBuffer.Deinterleave(outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpsampler->GetOrder());
      }
    }

    void PrepareBuffers(int numSamples)
    {
      int const oversamplingFactor = firUpsampler
                                       ? firUpsampler->GetOversamplingFactor()
                                       : (1 << iirUpsampler->GetOrder());
      int const numUpsampledSamples = numSamples * oversamplingFactor;
      outputBuffer.SetNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->PrepareBuffers(numSamples);
      }
      if (iirUpsampler) {
        iirOutputBuffer.SetNumSamples(numUpsampledSamples);
        iirUpsampler->PrepareBuffer(numSamples);
      }
    }

    ScalarToScalarUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<double>>(
          settings.numChannels, settings.firTransitionBand);
        firUpsampler->SetOversamplingFactor(1 << settings.order);
        iirUpsampler = nullptr;
        iirOutputBuffer.SetNumChannels(0);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<double>::New(settings.numChannels);
        iirUpsampler->SetOrder(settings.order);
        iirOutputBuffer.SetNumChannels(settings.numChannels);
        firUpsampler = nullptr;
      }
      PrepareBuffers(settings.numSamplesPerBlock);
    }

    int GetLatency()
    {
      if (firUpsampler) {
        return firUpsampler->GetNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int GetMaxUpsampledSamples()
    {
      if (firUpsampler) {
        return firUpsampler->GetMaxNumOutputSamples();
      }
      return 0;
    }

    void Reset()
    {
      if (firDownsampler) {
        firDownsampler->Reset();
      }
      if (iirDownsampler) {
        iirDownsampler->Reset();
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
        firDownsampler->SetOversamplingFactor(1 << settings.order);
        iirDownsampler = nullptr;
        firInputBuffer.SetNumChannels(settings.numChannels);
      }
      else {
        iirDownsampler =
          IirDownsamplerFactory<double>::New(settings.numChannels);
        iirDownsampler->SetOrder(settings.order);
        firInputBuffer.SetNumChannels(0);
        firDownsampler = nullptr;
      }
      PrepareBuffers(settings.numSamplesPerBlock,
                     settings.numSamplesPerBlock * (1 << settings.order));
    }

    void PrepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingFactor = firDownsampler
                                       ? firDownsampler->GetOversamplingFactor()
                                       : (1 << iirDownsampler->GetOrder());
      if (firDownsampler) {
        firDownsampler->PrepareBuffers(maxNumUpsampledSamples, numSamples);
        firInputBuffer.SetNumSamples(maxNumUpsampledSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->PrepareBuffer(numSamples);
      }
    }

    void ProcessBlock(InterleavedBuffer<Scalar>& input,
                      Scalar** output,
                      int numChannelsToUpsample,
                      int numSamples)
    {
      if (firDownsampler) {
        input.Deinterleave(firInputBuffer);
        firDownsampler->ProcessBlock(
          firInputBuffer, output, numChannelsToUpsample, numSamples);
      }
      else {
        iirDownsampler->ProcessBlock(
          input, numSamples * (1 << iirDownsampler->GetOrder()));
        iirDownsampler->GetOutput().Deinterleave(
          output, numChannelsToUpsample, numSamples);
      }
    }

    int GetLatency()
    {
      if (firDownsampler) {
        return (double)firDownsampler->GetNumSamplesBeforeOutputStarts() /
               (double)firDownsampler->GetOversamplingFactor();
      }
      return 0;
    }

    void Reset()
    {
      if (firDownsampler) {
        firDownsampler->Reset();
      }
      if (iirDownsampler) {
        iirDownsampler->Reset();
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
        firDownsampler->SetOversamplingFactor(1 << settings.order);
        iirDownsampler = nullptr;
        iirInputBuffer.SetNumChannels(0);
      }
      else {
        iirDownsampler =
          IirDownsamplerFactory<double>::New(settings.numChannels);
        iirDownsampler->SetOrder(settings.order);
        iirInputBuffer.SetNumChannels(2);
        firDownsampler = nullptr;
      }
      PrepareBuffers(settings.numSamplesPerBlock,
                     settings.numSamplesPerBlock * (1 << settings.order));
    }

    void PrepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingFactor = firDownsampler
                                       ? firDownsampler->GetOversamplingFactor()
                                       : (1 << iirDownsampler->GetOrder());
      if (firDownsampler) {
        firDownsampler->PrepareBuffers(maxNumUpsampledSamples, numSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->PrepareBuffer(numSamples);
        iirInputBuffer.SetNumSamples(maxNumUpsampledSamples);
      }
    }

    void ProcessBlock(Scalar** input,
                      Scalar** output,
                      int numChannelsToUpsample,
                      int numSamples)
    {
      if (firDownsampler) {
        firDownsampler->ProcessBlock(input,
                                     numChannelsToUpsample,
                                     output,
                                     numChannelsToUpsample,
                                     numSamples);
      }
      else {
        iirInputBuffer.Interleave(input, numChannelsToUpsample, numSamples);
        iirDownsampler->ProcessBlock(
          iirInputBuffer, numSamples * (1 << iirDownsampler->GetOrder()));
        iirDownsampler->GetOutput().Deinterleave(
          output, numChannelsToUpsample, numSamples);
      }
    }

    int GetLatency()
    {
      if (firDownsampler) {
        return (double)firDownsampler->GetNumSamplesBeforeOutputStarts() /
               (double)firDownsampler->GetOversamplingFactor();
      }
      return 0;
    }

    void Reset()
    {
      if (firDownsampler) {
        firDownsampler->Reset();
      }
      if (iirDownsampler) {
        iirDownsampler->Reset();
      }
    }
  };

  std::vector<std::unique_ptr<ScalarToVecUpsampler>> scalarToVecUpsamplers;

  std::vector<std::unique_ptr<VecToVecUpsampler>> vecToVecUpsamplers;

  std::vector<std::unique_ptr<ScalarToScalarUpsampler>>
    scalarToScalarUpsamplers;

  std::vector<std::unique_ptr<VecToScalarDownsampler>> vecToScalarDownsamplers;

  std::vector<std::unique_ptr<ScalarToScalarDownsampler>>
    scalarToScalarDownsamplers;

  std::vector<InterleavedBuffer<Scalar>> interleavedBuffers;

  std::vector<ScalarBuffer<Scalar>> scalarBuffers;

  // just a flag, usefull to trigger a reset on other objects when the
  // oversampling is changed
  bool isNew = true;

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
    for (int i = 0; i < settings.numScalarToScalarDownsamplers; ++i) {
      scalarToScalarDownsamplers.push_back(
        std::make_unique<ScalarToScalarDownsampler>(settings));
    }
    if (settings.UpdateLatency) {
      settings.UpdateLatency(GetLatency());
    }
    interleavedBuffers.resize(settings.numInterleavedBuffers);
    for (auto& buffer : interleavedBuffers) {
      buffer.SetNumChannels(settings.numChannels);
    }
    scalarBuffers.resize(settings.numScalarBuffers);
    for (auto& buffer : scalarBuffers) {
      buffer.SetNumChannels(settings.numChannels);
    }
  }

  void PrepareBuffers(int numSamples)
  {
    if (numSamplesPerBlock < numSamples) {

      numSamplesPerBlock = numSamples;

      for (auto& upsampler : scalarToVecUpsamplers) {
        upsampler->PrepareBuffers(numSamples);
      }
      for (auto& upsampler : vecToVecUpsamplers) {
        upsampler->PrepareBuffers(numSamples);
      }
      for (auto& upsampler : scalarToScalarUpsamplers) {
        upsampler->PrepareBuffers(numSamples);
      }

      int maxNumUpsampledSamples = rate * numSamples;

      if (scalarToVecUpsamplers.size() > 0) {
        maxNumUpsampledSamples =
          scalarToVecUpsamplers[0]->GetMaxUpsampledSamples();
      }
      else if (vecToVecUpsamplers.size() > 0) {
        maxNumUpsampledSamples =
          vecToVecUpsamplers[0]->GetMaxUpsampledSamples();
      }

      for (auto& downsampler : scalarToScalarDownsamplers) {
        downsampler->PrepareBuffers(numSamples, maxNumUpsampledSamples);
      }

      for (auto& buffer : interleavedBuffers) {
        buffer.Reserve(maxNumUpsampledSamples);
      }

      for (auto& buffer : scalarBuffers) {
        buffer.Reserve(maxNumUpsampledSamples);
      }
    }
  }

  int GetLatency()
  {
    int latency = 0;
    if (scalarToVecUpsamplers.size() > 0) {
      latency += scalarToVecUpsamplers[0]->GetLatency();
    }
    else if (vecToVecUpsamplers.size() > 0) {
      latency += vecToVecUpsamplers[0]->GetLatency();
    }
    if (vecToScalarDownsamplers.size() > 0) {
      latency += vecToScalarDownsamplers[0]->GetLatency();
    }
    else if (scalarToScalarDownsamplers.size() > 0) {
      latency += scalarToScalarDownsamplers[0]->GetLatency();
    }
    return latency;
  }

  void Reset()
  {
    for (auto& upsampler : scalarToVecUpsamplers) {
      upsampler->Reset();
    }
    for (auto& upsampler : vecToVecUpsamplers) {
      upsampler->Reset();
    }
    for (auto& downsampler : scalarToScalarDownsamplers) {
      downsampler->Reset();
    }
  }

  int GetRate() const { return rate; }

  int GetNumSamplesPerBlock() const { return numSamplesPerBlock; }
};

} // namespace oversimple
