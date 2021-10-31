/*
Copyright 2021 Dario Mambro

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
#include "FirOversampling.hpp"
#include "IirOversampling.hpp"

namespace oversimple {

enum class BufferType
{
  interleaved,
  plain
};

struct OversamplingSettings final
{
  uint32_t maxOrder = 5;
  uint32_t numDownSampledChannels = 2;
  uint32_t numUpSampledChannels = 2;
  uint32_t maxNumInputSamples = 128;
  BufferType upSampleOutputBufferType = BufferType::plain;
  BufferType upSampleInputBufferType = BufferType::plain;
  BufferType downSampleOutputBufferType = BufferType::plain;
  BufferType downSampleInputBufferType = BufferType::plain;
  uint32_t order = 1;
  bool isUsingLinearPhase = false;
  uint32_t fftBlockSize = 512;
  double firTransitionBand = 4.0;
};

template<class Float>
class Oversampling final
{
public:
  explicit Oversampling(OversamplingSettings settings)
    : settings{ settings }
    , iirUpSampler{ settings.numUpSampledChannels, settings.maxOrder }
    , iirDownSampler{ settings.numDownSampledChannels, settings.maxOrder }
    , firUpSampler{ settings.numUpSampledChannels, settings.maxOrder }
    , firDownSampler{ settings.numDownSampledChannels, settings.maxOrder }
  {
    setup();
    setOrder(settings.order);
  }

  void setMaxOrder(uint32_t value)
  {
    if (settings.maxOrder != value) {
      settings.maxOrder = value;
      setup();
    }
  }

  void setNumChannelsToUpSample(uint32_t numChannels)
  {
    if (settings.numUpSampledChannels != numChannels) {
      settings.numUpSampledChannels = numChannels;
      setup();
    }
  }

  void setNumChannelsToDownSample(uint32_t numChannels)
  {
    if (settings.numDownSampledChannels != numChannels) {
      settings.numDownSampledChannels = numChannels;
      setup();
    }
  }

  void prepareBuffers(uint32_t maxNumInputSamples_)
  {
    if (settings.maxNumInputSamples != maxNumInputSamples_) {
      settings.maxNumInputSamples = maxNumInputSamples_;
      prepareInternalBuffers();
    }
  }

  void setFirFftBlockSize(uint32_t value)
  {
    if (settings.fftBlockSize != value) {
      settings.fftBlockSize = value;
      setup();
    }
  }

  void setFirTransitionBand(double transitionBand)
  {
    if (settings.firTransitionBand != transitionBand) {
      settings.firTransitionBand = transitionBand;
      setup();
    }
  }

  void setUseLinearPhase(bool useLinearPhase)
  {
    if (!settings.isUsingLinearPhase != useLinearPhase) {
      settings.isUsingLinearPhase = useLinearPhase;
      reset();
    }
  }

  void setOrder(uint32_t order)
  {
    settings.order = order;
    firUpSampler.setOrder(order);
    iirUpSampler.setOrder(order);
    firDownSampler.setOrder(order);
    iirDownSampler.setOrder(order);
  }

  void reset()
  {
    if (settings.isUsingLinearPhase) {
      firUpSampler.reset();
      iirUpSampler.reset();
    }
    else {
      firDownSampler.reset();
      iirDownSampler.reset();
    }
  }

  uint32_t getUpSamplingLatency()
  {
    if (settings.isUsingLinearPhase) {
      if (settings.numUpSampledChannels > 0) {
        return firUpSampler.getNumSamplesBeforeOutputStarts();
      }
    }
    return 0;
  }

  uint32_t getDownSamplingLatency()
  {
    if (settings.isUsingLinearPhase) {
      if (settings.numDownSampledChannels > 0) {
        return firDownSampler.getNumSamplesBeforeOutputStarts();
      }
    }
    return 0;
  }

  uint32_t getLatency()
  {
    if (settings.isUsingLinearPhase) {
      return getUpSamplingLatency() + getDownSamplingLatency() / getOversamplingRate();
    }
    return 0;
  }

  uint32_t getMaxNumOutputSamples()
  {
    if (settings.isUsingLinearPhase) {
      if (settings.numUpSampledChannels > 0) {
        return firUpSampler.getMaxNumOutputSamples();
      }
    }
    return settings.maxNumInputSamples * getOversamplingRate();
  }

  uint32_t getOversamplingOrder() const
  {
    return settings.order;
  }

  uint32_t getOversamplingRate() const
  {
    return 1 << settings.order;
  }

  void setUpSampledOutputBufferType(BufferType bufferType)
  {
    settings.upSampleOutputBufferType = bufferType;
  }

  void setDownSampledOutputBufferType(BufferType bufferType)
  {
    settings.downSampleOutputBufferType = bufferType;
  }

  void setDownSampledInputBufferType(BufferType bufferType)
  {
    settings.downSampleInputBufferType = bufferType;
  }

  uint32_t upSample(Float* const* input, uint32_t numSamples)
  {
    assert(settings.upSampleInputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase) {
      auto const numUpSampledSamples = firUpSampler.processBlock(input, numSamples);
      if (settings.upSampleOutputBufferType == BufferType::interleaved) {
        assert(firUpSampler.getOutput().getNumSamples() == numUpSampledSamples);
        assert(upSampleOutputInterleaved.getCapacity() >= numUpSampledSamples);
        upSampleOutputInterleaved.setNumSamples(numUpSampledSamples);
        upSampleOutputInterleaved.interleave(firUpSampler.getOutput());
      }
      return numUpSampledSamples;
    }
    else {
      iirUpSampler.processBlock(input, numSamples);
      auto const numUpSampledSamples = numSamples * getOversamplingRate();
      if (settings.upSampleOutputBufferType == BufferType::plain) {
        assert(numUpSampledSamples == iirUpSampler.getOutput().getNumSamples());
        assert(upSamplePlainBuffer.getCapacity() >= numUpSampledSamples);
        upSamplePlainBuffer.setNumSamples(numUpSampledSamples);
        iirUpSampler.getOutput().deinterleave(upSamplePlainBuffer);
      }
      return numUpSampledSamples;
    }
  }

  uint32_t upSample(InterleavedBuffer<Float> const& input)
  {
    assert(settings.upSampleInputBufferType == BufferType::interleaved);
    assert(input.getNumChannels() == settings.numUpSampledChannels);
    if (settings.isUsingLinearPhase) {
      assert(upSamplePlainBuffer.getCapacity() >= input.getNumSamples());
      upSamplePlainBuffer.setNumSamples(input.getNumSamples());
      input.deinterleave(upSamplePlainBuffer);
      auto const numUpSampledSamples = firUpSampler.processBlock(upSamplePlainBuffer);
      if (settings.upSampleOutputBufferType == BufferType::interleaved) {
        assert(firUpSampler.getOutput() == numUpSampledSamples);
        assert(upSampleOutputInterleaved.getCapacity() >= numUpSampledSamples);
        upSampleOutputInterleaved.setNumSamples(numUpSampledSamples);
        upSampleOutputInterleaved.interleave(firUpSampler.getOutput());
      }
      return numUpSampledSamples;
    }
    else {
      auto const numUpSampledSamples = iirUpSampler.processBlock(input);
      if (settings.upSampleOutputBufferType == BufferType::plain) {
        assert(numUpSampledSamples == iirUpSampler.getOutput().getNumSamples());
        assert(upSamplePlainBuffer.getCapacity() >= numUpSampledSamples);
        upSamplePlainBuffer.setNumSamples(numUpSampledSamples);
        iirUpSampler.getOutput().deinterleave(upSamplePlainBuffer);
      }
      return numUpSampledSamples;
    }
  }

  InterleavedBuffer<Float>& getUpSampleOutputInterleaved()
  {
    assert(settings.upSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return upSampleOutputInterleaved;
    else
      return iirUpSampler.getOutput();
  }

  Buffer<Float>& getUpSampleOutput()
  {
    assert(settings.upSampleOutputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase)
      return firUpSampler.getOutput();
    return upSamplePlainBuffer;
  }

  InterleavedBuffer<Float> const& getUpSampleOutputInterleaved() const
  {
    assert(settings.upSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return upSampleOutputInterleaved;
    else
      return iirUpSampler.getOutput();
  }

  Buffer<Float> const& getUpSampleOutput() const
  {
    assert(settings.upSampleOutputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase)
      return firUpSampler.getOutput();
    return upSamplePlainBuffer;
  }

  void downSample(Float* const* input, uint32_t numInputSamples, Float** output, uint32_t numOutputSamples)
  {
    assert(settings.downSampleOutputBufferType == BufferType::plain);
    assert(settings.downSampleInputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase) {
      firDownSampler.processBlock(input, numInputSamples, output, numOutputSamples);
    }
    else {
      assert(numOutputSamples == numInputSamples * (1 << settings.order));
      assert(downSampleBufferInterleaved.getCapacity() >= numInputSamples);
      downSampleBufferInterleaved.setNumSamples(numInputSamples);
      downSampleBufferInterleaved.interleave(input, settings.numDownSampledChannels, numInputSamples);
      iirDownSampler.processBlock(downSampleBufferInterleaved);
      iirDownSampler.getOutput().deinterleave(output, settings.numDownSampledChannels, numOutputSamples);
    }
  }

  void downSample(InterleavedBuffer<Float> const& input, Float** output, uint32_t numOutputSamples)
  {
    assert(settings.downSampleOutputBufferType == BufferType::plain);
    assert(settings.downSampleInputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase) {
      auto const numInputSamples = input.getNumSamples();
      assert(downSamplePlainInputBuffer.getCapacity() >= numInputSamples);
      downSamplePlainInputBuffer.setNumSamples(numInputSamples);
      input.deinterleave(downSamplePlainInputBuffer);
      firDownSampler.processBlock(downSamplePlainInputBuffer, output, numOutputSamples);
    }
    else {
      assert(numOutputSamples == input.getNumSamples() * (1 << settings.order));
      iirDownSampler.processBlock(input);
      iirDownSampler.getOutput().deinterleave(output, numOutputSamples);
    }
  }

  void downSample(Float* const* input, uint32_t numInputSamples, uint32_t numOutputSamples)
  {
    assert(settings.downSampleOutputBufferType == BufferType::interleaved);
    assert(settings.downSampleInputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase) {
      assert(downSamplePlainOutputBuffer.getCapacity() >= numOutputSamples);
      assert(downSampleBufferInterleaved.getCapacity() >= numOutputSamples);
      downSamplePlainOutputBuffer.setNumSamples(numOutputSamples);
      downSampleBufferInterleaved.setNumSamples(numOutputSamples);
      firDownSampler.processBlock(input, numInputSamples, downSamplePlainOutputBuffer.get(), numOutputSamples);
      downSampleBufferInterleaved.interleave(downSamplePlainOutputBuffer, numOutputSamples);
    }
    else {
      assert(numOutputSamples == numInputSamples * (1 << settings.order));
      assert(downSampleBufferInterleaved.getCapacity() >= numOutputSamples);
      downSampleBufferInterleaved.setNumSamples(numInputSamples);
      downSampleBufferInterleaved.interleave(input, numInputSamples);
      iirDownSampler.processBlock(downSampleBufferInterleaved);
    }
  }

  void downSample(InterleavedBuffer<Float> const& input, uint32_t numOutputSamples)
  {
    assert(settings.downSampleOutputBufferType == BufferType::interleaved);
    assert(settings.downSampleInputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase) {
      assert(downSamplePlainInputBuffer.getCapacity() >= input.getNumSamples());
      assert(downSamplePlainOutputBuffer.getCapacity() >= numOutputSamples);
      downSamplePlainInputBuffer.setNumSamples(input.getNumSamples());
      downSamplePlainOutputBuffer.setNumSamples(numOutputSamples);
      input.deinterleave(downSamplePlainInputBuffer);
      firDownSampler.processBlock(downSamplePlainInputBuffer, downSamplePlainOutputBuffer, numOutputSamples);
      downSampleBufferInterleaved.interleave(downSamplePlainOutputBuffer, numOutputSamples);
    }
    else {
      assert(numOutputSamples == input.getNumSamples() * (1 << settings.order));
      iirDownSampler.processBlock(input);
    }
  }

  InterleavedBuffer<Float>& getDownSampleOutputInterleaved()
  {
    assert(settings.downSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return downSampleBufferInterleaved;
    else
      iirDownSampler.getOutput();
  }

private:
  void setup()
  {
    iirUpSampler.setNumChannels(settings.numUpSampledChannels);
    iirUpSampler.setMaxOrder(settings.maxOrder);

    iirDownSampler.setNumChannels(settings.numDownSampledChannels);
    iirDownSampler.setMaxOrder(settings.maxOrder);

    firUpSampler.setTransitionBand(settings.firTransitionBand);
    firUpSampler.setFftSamplesPerBlock(settings.fftBlockSize);
    firUpSampler.setNumChannels(settings.numUpSampledChannels);
    firUpSampler.setMaxOrder(settings.maxOrder);

    firDownSampler.setTransitionBand(settings.firTransitionBand);
    firDownSampler.setFftSamplesPerBlock(settings.fftBlockSize);
    firDownSampler.setNumChannels(settings.numDownSampledChannels);
    firDownSampler.setMaxOrder(settings.maxOrder);

    setupInputOutputBuffers();
    prepareInternalBuffers();
  }

  void prepareInternalBuffers()
  {
    iirUpSampler.prepareBuffers(settings.maxNumInputSamples);
    iirDownSampler.prepareBuffers(settings.maxNumInputSamples);
    firUpSampler.prepareBuffers(settings.maxNumInputSamples);
    auto const maxFirUpSampledSamples = firUpSampler.getMaxNumOutputSamples();
    firDownSampler.prepareBuffers(maxFirUpSampledSamples, settings.maxNumInputSamples);
    auto const maxSamplesUpSampled = settings.maxNumInputSamples * (1 << settings.maxOrder);
    auto const maxSamples = std::max(maxFirUpSampledSamples, maxSamplesUpSampled);
    downSampleBufferInterleaved.reserve(maxSamples);
    downSamplePlainOutputBuffer.reserve(maxSamples);
    downSamplePlainInputBuffer.reserve(maxSamples);
    upSampleOutputInterleaved.reserve(maxSamples);
    upSamplePlainBuffer.reserve(maxSamples);
  }

  void setupInputOutputBuffers()
  {
    if (settings.upSampleOutputBufferType == BufferType::interleaved) {
      upSampleOutputInterleaved.setNumChannels(settings.numUpSampledChannels);
    }
    else {
      upSampleOutputInterleaved.setNumChannels(0);
    }
    if (settings.upSampleOutputBufferType == BufferType::plain ||
        settings.upSampleInputBufferType == BufferType::interleaved)
    {
      upSamplePlainBuffer.setNumChannels(settings.numUpSampledChannels);
    }
    else {
      upSamplePlainBuffer.setNumChannels(0);
    }
    if (settings.downSampleOutputBufferType == BufferType::plain &&
        settings.downSampleInputBufferType == BufferType::plain) {
      downSampleBufferInterleaved.setNumChannels(settings.numDownSampledChannels);
      downSamplePlainInputBuffer.setNumChannels(0);
      downSamplePlainOutputBuffer.setNumChannels(0);
    }
    else if (settings.downSampleOutputBufferType == BufferType::interleaved &&
             settings.downSampleInputBufferType == BufferType::interleaved)
    {
      downSampleBufferInterleaved.setNumChannels(0);
      downSamplePlainInputBuffer.setNumChannels(0);
      downSamplePlainOutputBuffer.setNumChannels(settings.numDownSampledChannels);
    }
    else if (settings.downSampleOutputBufferType == BufferType::plain &&
             settings.downSampleInputBufferType == BufferType::interleaved)
    {
      downSampleBufferInterleaved.setNumChannels(0);
      downSamplePlainInputBuffer.setNumChannels(settings.numDownSampledChannels);
      downSamplePlainOutputBuffer.setNumChannels(settings.numDownSampledChannels);
    }
    else if (settings.downSampleOutputBufferType == BufferType::interleaved &&
             settings.downSampleInputBufferType == BufferType::plain)
    {
      downSampleBufferInterleaved.setNumChannels(settings.numDownSampledChannels);
      downSamplePlainInputBuffer.setNumChannels(0);
      downSamplePlainOutputBuffer.setNumChannels(settings.numDownSampledChannels);
    }
  }

  OversamplingSettings settings;

  fir::TUpSamplerPreAllocated<Float> firUpSampler;
  fir::TDownSamplerPreAllocated<Float> firDownSampler;
  iir::UpSampler<Float> iirUpSampler;
  iir::DownSampler<Float> iirDownSampler;

  InterleavedBuffer<Float> downSampleBufferInterleaved;
  Buffer<Float> downSamplePlainOutputBuffer;
  Buffer<Float> downSamplePlainInputBuffer;
  InterleavedBuffer<Float> upSampleOutputInterleaved;
  Buffer<Float> upSamplePlainBuffer;
};

} // namespace oversimple