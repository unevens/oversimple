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

struct OversamplingSettings
{
  int numChannels;

  int numUpsamplers;
  int numInterleavedUpsamplers;
  int numDownsamplers;
  int numInterleavedDownsamplers;

  int numBuffers;
  int numInterleavedBuffers;

  int order;
  bool linearPhase;
  int numSamplesPerBlock;
  double firTransitionBand;

  std::function<void(int)> UpdateLatency;

  OversamplingSettings(std::function<void(int)> UpdateLatency,
                       int numChannels = 2,
                       int numUpsamplers = 1,
                       int numDownsamplers = 1,
                       int numInterleavedUpsamplers = 0,
                       int numInterleavedDownsamplers = 0,
                       double firTransitionBand = 4.0,
                       int order = 1,
                       bool linearPhase = false,
                       int numSamplesPerBlock = 256)
    : UpdateLatency(UpdateLatency)
    , numChannels(numChannels)
    , numUpsamplers(numUpsamplers)
    , numInterleavedUpsamplers(numInterleavedUpsamplers)
    , numDownsamplers(numDownsamplers)
    , numInterleavedDownsamplers(numInterleavedDownsamplers)
    , order(order)
    , linearPhase(linearPhase)
    , numSamplesPerBlock(numSamplesPerBlock)
    , firTransitionBand(firTransitionBand)
    , numBuffers(numBuffers)
    , numInterleavedBuffers(numInterleavedBuffers)
  {}
};

template<typename Scalar>
class Oversampling
{
  int numSamplesPerBlock;
  int rate;

public:
  class InterleavedUpsampler
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
        firInputBuffer.SetSize(numSamples);
        firOutputBuffer.SetSize(numUpsampledSamples);
      }
      if (iirUpsampler) {
        iirUpsampler->PrepareBuffer(numSamples);
      }
    }

    InterleavedUpsampler(OversamplingSettings const& settings)
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

  class Upsampler
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
      firOutputBuffer.SetSize(numUpsampledSamples);
      outputBuffer.SetNumSamples(numUpsampledSamples);
      if (firUpsampler) {
        firUpsampler->PrepareBuffers(numSamples);
      }
      if (iirUpsampler) {
        iirUpsampler->PrepareBuffer(numSamples);
      }
    }

    Upsampler(OversamplingSettings const& settings)
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

  struct InterleavedDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    ScalarBuffer<Scalar> firInputBuffer;

  public:
    InterleavedDownsampler(OversamplingSettings const& settings)
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
        firInputBuffer.SetSize(maxNumUpsampledSamples);
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

  struct Downsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    InterleavedBuffer<Scalar> iirInputBuffer;

  public:
    Downsampler(OversamplingSettings const& settings)
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

  std::vector<std::unique_ptr<Upsampler>> upsamplers;

  std::vector<std::unique_ptr<InterleavedUpsampler>> interleavedUpsamplers;

  std::vector<std::unique_ptr<InterleavedDownsampler>> interleavedDownsamplers;

  std::vector<std::unique_ptr<Downsampler>> downsamplers;

  std::vector<InterleavedBuffer<Scalar>> interleavedBuffers;

  std::vector<ScalarBuffer<Scalar>> buffers;

  bool isNew = true;

  Oversampling(OversamplingSettings const& settings)
    : numSamplesPerBlock(settings.numSamplesPerBlock)
    , rate(1 << settings.order)
  {
    for (int i = 0; i < settings.numUpsamplers; ++i) {
      upsamplers.push_back(std::make_unique<Upsampler>(settings));
    }
    for (int i = 0; i < settings.numInterleavedUpsamplers; ++i) {
      interleavedUpsamplers.push_back(
        std::make_unique<InterleavedUpsampler>(settings));
    }
    for (int i = 0; i < settings.numInterleavedDownsamplers; ++i) {
      interleavedDownsamplers.push_back(
        std::make_unique<InterleavedDownsampler>(settings));
    }
    for (int i = 0; i < settings.numDownsamplers; ++i) {
      downsamplers.push_back(std::make_unique<Downsampler>(settings));
    }
    if (settings.UpdateLatency) {
      settings.UpdateLatency(GetLatency());
    }
    interleavedBuffers.resize(settings.numInterleavedBuffers);
    for (auto& buffer : interleavedBuffers) {
      buffer.SetNumChannels(settings.numChannels);
    }
    buffers.resize(settings.numBuffers);
    for (auto& buffer : buffers) {
      buffer.SetNumChannels(settings.numChannels);
    }
  }

  void PrepareBuffers(int numSamples)
  {
    if (numSamplesPerBlock < numSamples) {
      numSamplesPerBlock = numSamples;
      for (auto& upsampler : upsamplers) {
        upsampler->PrepareBuffers(numSamples);
      }
      for (auto& upsampler : interleavedUpsamplers) {
        upsampler->PrepareBuffers(numSamples);
      }

      int maxNumUpsampledSamples = rate * numSamples;

      if (upsamplers.size() > 0) {
        maxNumUpsampledSamples = upsamplers[0]->GetMaxUpsampledSamples();
      }
      else if (interleavedUpsamplers.size() > 0) {
        maxNumUpsampledSamples =
          interleavedUpsamplers[0]->GetMaxUpsampledSamples();
      }

      for (auto& downsampler : downsamplers) {
        downsampler->PrepareBuffers(numSamples, maxNumUpsampledSamples);
      }

      for (auto& buffer : interleavedBuffers) {
        buffer.Reserve(maxNumUpsampledSamples);
      }

      for (auto& buffer : buffers) {
        buffer.Reserve(maxNumUpsampledSamples);
      }
    }
  }

  int GetLatency()
  {
    int latency = 0;
    if (upsamplers.size() > 0) {
      latency += upsamplers[0]->GetLatency();
    }
    else if (interleavedUpsamplers.size() > 0) {
      latency += interleavedUpsamplers[0]->GetLatency();
    }
    if (downsamplers.size() > 0) {
      latency += downsamplers[0]->GetLatency();
    }
    return latency;
  }

  void Reset()
  {
    for (auto& upsampler : upsamplers) {
      upsampler->Reset();
    }
    for (auto& upsampler : interleavedUpsamplers) {
      upsampler->Reset();
    }
    for (auto& downsampler : downsamplers) {
      downsampler->Reset();
    }
  }

  int GetRate() const { return rate; }

  int GetNumSamplesPerBlock() const { return numSamplesPerBlock; }
};

} // namespace oversimple
