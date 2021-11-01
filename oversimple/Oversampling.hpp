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

/**
 * enumeration used to refer to either a Buffer object, which is a plain buffer, meaning that the samples of each
 * channels are contiguously stored; or an InterleavedBuffer object, in which the samples of all channels are
 * interleaved.
 * */
enum class BufferType
{
  interleaved,
  plain
};

/*
 * A struct that contains all settings needed to specify the behaviour of an Oversampling object.
 * */
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

/*
 * A class that wraps the FIR and IIR re-samplers and supports both plain and interleaved buffers and inputs/outputs.
 * */
template<class Float>
class TOversampling final
{
public:
  /**
   * Constructor
   * @param settings the settings to initialize the object with.
   * */
  explicit TOversampling(OversamplingSettings settings)
    : settings{ settings }
    , iirUpSampler{ settings.numUpSampledChannels, settings.maxOrder }
    , iirDownSampler{ settings.numDownSampledChannels, settings.maxOrder }
    , firUpSampler{ settings.numUpSampledChannels, settings.maxOrder }
    , firDownSampler{ settings.numDownSampledChannels, settings.maxOrder }
  {
    setup();
    setOrder(settings.order);
  }

  /**
   * @return the current settings of the object
   */
  OversamplingSettings const& getSettings() const
  {
    return settings;
  }

  /**
   * Sets the maximum order of oversampling supported, and allocates the necessary resources
   * @param value the maximum order of oversampling
   */
  void setMaxOrder(uint32_t value)
  {
    if (settings.maxOrder != value) {
      settings.maxOrder = value;
      setup();
    }
  }

  /**
   * Prepare the up-samplers to work with the supplied number of channels.
   * @param numChannels the new number of channels to prepare the up-samplers for.
   */
  void setNumChannelsToUpSample(uint32_t numChannels)
  {
    if (settings.numUpSampledChannels != numChannels) {
      settings.numUpSampledChannels = numChannels;
      setup();
    }
  }

  /**
   * Prepare the down-samplers to work with the supplied number of channels.
   * @param numChannels the new number of channels to prepare the down-samplers for.
   */
  void setNumChannelsToDownSample(uint32_t numChannels)
  {
    if (settings.numDownSampledChannels != numChannels) {
      settings.numDownSampledChannels = numChannels;
      setup();
    }
  }

  /**
   * Allocates resources to process up to maxNumInputSamples input.
   * @param maxNumInputSamples the expected maximum number input samples
   */
  void prepareBuffers(uint32_t maxNumInputSamples)
  {
    if (settings.maxNumInputSamples != maxNumInputSamples) {
      settings.maxNumInputSamples = maxNumInputSamples;
      prepareInternalBuffers();
    }
  }

  /**
   * Sets the number of samples that are processed by each fft call. It only affects the FIR re-samplers used when
   * linear phase is enabled.
   * @param value the new number of samples that will be processed by each fft call.
   */
  void setFirFftBlockSize(uint32_t value)
  {
    if (settings.fftBlockSize != value) {
      settings.fftBlockSize = value;
      setup();
    }
  }

  /**
   * Sets the antialiasing fir filter transition band. Only affects the behaviour of the object when linear phase is
   * enabled.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setFirTransitionBand(double transitionBand)
  {
    if (settings.firTransitionBand != transitionBand) {
      settings.firTransitionBand = transitionBand;
      setup();
    }
  }

  /**
   * Sets whether the object shoul use the linear phase FIR re-samplers or the minimum-phase IIR re-samplers.
   * @param useLinearPhase true to enable linear phase, false to disable it.
   * */
  void setUseLinearPhase(bool useLinearPhase)
  {
    if (!settings.isUsingLinearPhase != useLinearPhase) {
      settings.isUsingLinearPhase = useLinearPhase;
      reset();
    }
  }

  /**
   * Sets the order of oversampling to be used. It must be less or equal to the maximum order set
   * @value the order to set
   * @return true if the order was set correctly, false otherwise
   */
  void setOrder(uint32_t order)
  {
    assert(order > 0 && order <= 5);
    settings.order = order;
    firUpSampler.setOrder(order);
    iirUpSampler.setOrder(order);
    firDownSampler.setOrder(order);
    iirDownSampler.setOrder(order);
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
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

  /**
   * @return the number of input samples to the up-sampling call needed before a first output sample is
   * produced by the up-sampling call.
   */
  uint32_t getUpSamplingLatency()
  {
    if (settings.isUsingLinearPhase) {
      if (settings.numUpSampledChannels > 0) {
        return firUpSampler.getNumSamplesBeforeOutputStarts();
      }
    }
    return 0;
  }

  /**
   * @return the number of input samples to the down-sampling call needed before a first output sample is
   * produced by the down-sampling call.
   */
  uint32_t getDownSamplingLatency()
  {
    if (settings.isUsingLinearPhase) {
      if (settings.numDownSampledChannels > 0) {
        return firDownSampler.getNumSamplesBeforeOutputStarts();
      }
    }
    return 0;
  }

  /**
   * @return the number of input samples to the up-sampling call needed before a first output sample is
   * produced by the down-sampling call.
   */
  uint32_t getLatency()
  {
    if (settings.isUsingLinearPhase) {
      return getUpSamplingLatency() + getDownSamplingLatency() / getOversamplingRate();
    }
    return 0;
  }

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers.
   */
  uint32_t getMaxNumOutputSamples()
  {
    if (settings.isUsingLinearPhase) {
      if (settings.numUpSampledChannels > 0) {
        return firUpSampler.getMaxNumOutputSamples();
      }
    }
    return settings.maxNumInputSamples * getOversamplingRate();
  }

  /**
   * @return the oversampling order currently in use.
   */
  uint32_t getOversamplingOrder() const
  {
    return settings.order;
  }

  /**
   * @return the oversampling rate currently in use.
   */
  uint32_t getOversamplingRate() const
  {
    return 1 << settings.order;
  }

  /**
   * Sets the type of output for the up-sampling.
   * @param bufferType the type of output for the up-sampling.
   */
  void setUpSampledOutputBufferType(BufferType bufferType)
  {
    settings.upSampleOutputBufferType = bufferType;
  }

  /**
   * Sets the type of output for the down-sampling.
   * @param bufferType the type of output for the down-sampling.
   */
  void setDownSampledOutputBufferType(BufferType bufferType)
  {
    settings.downSampleOutputBufferType = bufferType;
  }

  /**
   * Sets the type of input for the down-sampling.
   * @param bufferType the type of input for the down-sampling.
   */
  void setDownSampledInputBufferType(BufferType bufferType)
  {
    settings.downSampleInputBufferType = bufferType;
  }

  /**
   * Up-samples the input to a buffer owned by the object.
   * @param input pointer to the input buffers
   * @param numSamples the number of samples in each channel of the input buffer
   */
  uint32_t upSample(Float* const* input, uint32_t numSamples)
  {
    assert(settings.upSampleInputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase) {
      auto const numUpSampledSamples = firUpSampler.processBlock(input, numSamples);
      if (settings.upSampleOutputBufferType == BufferType::interleaved) {
        assert(firUpSampler.getOutput().getNumSamples() == numUpSampledSamples);
        assert(upSampleOutputInterleaved.getCapacity() >= numUpSampledSamples);
        upSampleOutputInterleaved.setNumSamples(numUpSampledSamples);
        bool const ok = upSampleOutputInterleaved.interleave(firUpSampler.getOutput());
        assert(ok);
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

  /**
   * Up-samples the input to a buffer owned by the object.
   * @param input pointer to the input buffers
   */
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
        assert(firUpSampler.getOutput().getNumSamples() == numUpSampledSamples);
        assert(upSampleOutputInterleaved.getCapacity() >= numUpSampledSamples);
        upSampleOutputInterleaved.setNumSamples(numUpSampledSamples);
        bool const ok = upSampleOutputInterleaved.interleave(firUpSampler.getOutput());
        assert(ok);
      }
      return numUpSampledSamples;
    }
    else {
      iirUpSampler.processBlock(input);
      auto const numUpSampledSamples = iirUpSampler.getOutput().getNumSamples();
      if (settings.upSampleOutputBufferType == BufferType::plain) {
        assert(upSamplePlainBuffer.getCapacity() >= numUpSampledSamples);
        upSamplePlainBuffer.setNumSamples(numUpSampledSamples);
        iirUpSampler.getOutput().deinterleave(upSamplePlainBuffer);
      }
      return numUpSampledSamples;
    }
  }

  /**
   * @return an interleaved buffer that holds the output of the up-sampling.
   */
  InterleavedBuffer<Float>& getUpSampleOutputInterleaved()
  {
    assert(settings.upSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return upSampleOutputInterleaved;
    else
      return iirUpSampler.getOutput();
  }

  /**
   * @return a buffer that holds the output of the up-sampling.
   */
  Buffer<Float>& getUpSampleOutput()
  {
    assert(settings.upSampleOutputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase)
      return firUpSampler.getOutput();
    return upSamplePlainBuffer;
  }

  /**
   * @return a const interleaved buffer that holds the output of the up-sampling.
   */
  InterleavedBuffer<Float> const& getUpSampleOutputInterleaved() const
  {
    assert(settings.upSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return upSampleOutputInterleaved;
    else
      return iirUpSampler.getOutput();
  }

  /**
   * @return a const buffer that holds the output of the up-sampling.
   */
  Buffer<Float> const& getUpSampleOutput() const
  {
    assert(settings.upSampleOutputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase)
      return firUpSampler.getOutput();
    return upSamplePlainBuffer;
  }

  /**
   * Down-samples the input.
   * @param input pointer to the input buffer.
   * @param numInputSamples the number of samples of each channel of the input
   * buffer.
   * @param output pointer to the memory in which to store the down-sampled data.
   * @param numOutputSamples the number of samples needed as output
   */
  void downSample(Float* const* input, uint32_t numInputSamples, Float** output, uint32_t numOutputSamples)
  {
    assert(settings.downSampleOutputBufferType == BufferType::plain);
    assert(settings.downSampleInputBufferType == BufferType::plain);
    if (settings.isUsingLinearPhase) {
      firDownSampler.processBlock(input, numInputSamples, output, numOutputSamples);
    }
    else {
      assert(numOutputSamples * (1 << settings.order) == numInputSamples);
      assert(downSampleBufferInterleaved.getCapacity() >= numInputSamples);
      downSampleBufferInterleaved.setNumSamples(numInputSamples);
      bool const ok = downSampleBufferInterleaved.interleave(input, settings.numDownSampledChannels, numInputSamples);
      assert(ok);
      iirDownSampler.processBlock(downSampleBufferInterleaved);
      iirDownSampler.getOutput().deinterleave(output, settings.numDownSampledChannels, numOutputSamples);
    }
  }

  /**
   * Down-samples the input.
   * @param input an interleaved buffer holding the input.
   * @param output pointer to the memory in which to store the down-sampled data.
   * @param numOutputSamples the number of samples needed as output
   */
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
      assert(numOutputSamples * (1 << settings.order) == input.getNumSamples());
      iirDownSampler.processBlock(input);
      iirDownSampler.getOutput().deinterleave(output, settings.numDownSampledChannels, numOutputSamples);
    }
  }

  /**
   * Down-samples the input to an InterleavedBuffer owned by the object.
   * @param input pointer to the input buffer.
   * @param numInputSamples the number of samples of each channel of the input
   * buffer.
   * @param numOutputSamples the number of samples needed as output
   */
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
      downSampleBufferInterleaved.setNumSamples(numOutputSamples);
      bool const ok = downSampleBufferInterleaved.interleave(downSamplePlainOutputBuffer);
      assert(ok);
    }
    else {
      assert(numOutputSamples * (1 << settings.order) == numInputSamples);
      assert(downSampleBufferInterleaved.getCapacity() >= numOutputSamples);
      downSampleBufferInterleaved.setNumSamples(numInputSamples);
      bool const ok = downSampleBufferInterleaved.interleave(input, settings.numDownSampledChannels, numInputSamples);
      assert(ok);
      iirDownSampler.processBlock(downSampleBufferInterleaved);
    }
  }

  /**
   * Down-samples the input to an InterleavedBuffer owned by the object.
   * @param input an interleaved buffer holding the input.
   * @param numOutputSamples the number of samples needed as output
   */
  void downSample(InterleavedBuffer<Float> const& input, uint32_t numOutputSamples)
  {
    assert(settings.downSampleOutputBufferType == BufferType::interleaved);
    assert(settings.downSampleInputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase) {
      assert(downSamplePlainInputBuffer.getCapacity() >= input.getNumSamples());
      assert(downSamplePlainOutputBuffer.getCapacity() >= numOutputSamples);
      assert(downSampleBufferInterleaved.getCapacity() >= numOutputSamples);
      downSamplePlainInputBuffer.setNumSamples(input.getNumSamples());
      downSamplePlainOutputBuffer.setNumSamples(numOutputSamples);
      input.deinterleave(downSamplePlainInputBuffer);
      firDownSampler.processBlock(downSamplePlainInputBuffer, downSamplePlainOutputBuffer, numOutputSamples);
      downSampleBufferInterleaved.setNumSamples(numOutputSamples);
      bool const ok = downSampleBufferInterleaved.interleave(downSamplePlainOutputBuffer);
      assert(ok);
    }
    else {
      assert(numOutputSamples * (1 << settings.order) == input.getNumSamples());
      iirDownSampler.processBlock(input);
    }
  }

  /**
   * @return an interleaved buffer that holds the output of the down-sampling.
   */
  InterleavedBuffer<Float>& getDownSampleOutputInterleaved()
  {
    assert(settings.downSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return downSampleBufferInterleaved;
    else
      return iirDownSampler.getOutput();
  }

  /**
   * @return a const interleaved buffer that holds the output of the down-sampling.
   */
  InterleavedBuffer<Float> const& getDownSampleOutputInterleaved() const
  {
    assert(settings.downSampleOutputBufferType == BufferType::interleaved);
    if (settings.isUsingLinearPhase)
      return downSampleBufferInterleaved;
    else
      return iirDownSampler.getOutput();
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
      downSampleBufferInterleaved.setNumChannels(settings.numDownSampledChannels);
      downSamplePlainInputBuffer.setNumChannels(settings.numDownSampledChannels);
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

/*
 * A class that wraps a TOversampling<float> and a TOversampling<double>
 * */
class Oversampling final
{
public:
  /**
   * Constructor
   * @param settings the settings to initialize the object with.
   * */
  explicit Oversampling(OversamplingSettings settings)
    : oversampling32{ settings }
    , oversampling64{ settings }
  {}

  /**
   * @return the current settings of the object
   */
  OversamplingSettings const& getSettings() const
  {
    return oversampling32.getSettings();
  }

  /**
   * Sets the maximum order of oversampling supported, and allocates the necessary resources
   * @param value the maximum order of oversampling
   */
  void setMaxOrder(uint32_t value)
  {
    oversampling32.setMaxOrder(value);
    oversampling64.setMaxOrder(value);
  }

  /**
   * Prepare the up-samplers to work with the supplied number of channels.
   * @param numChannels the new number of channels to prepare the up-samplers for.
   */
  void setNumChannelsToUpSample(uint32_t numChannels)
  {
    oversampling32.setNumChannelsToUpSample(numChannels);
    oversampling64.setNumChannelsToUpSample(numChannels);
  }

  /**
   * Prepare the down-samplers to work with the supplied number of channels.
   * @param numChannels the new number of channels to prepare the down-samplers for.
   */
  void setNumChannelsToDownSample(uint32_t numChannels)
  {
    oversampling32.setNumChannelsToDownSample(numChannels);
    oversampling64.setNumChannelsToDownSample(numChannels);
  }

  /**
   * Allocates resources to process up to maxNumInputSamples input.
   * @param maxNumInputSamples the expected maximum number input samples
   */
  void prepareBuffers(uint32_t maxNumInputSamples)
  {
    oversampling32.prepareBuffers(maxNumInputSamples);
    oversampling64.prepareBuffers(maxNumInputSamples);
  }

  /**
   * Sets the number of samples that are processed by each fft call. It only affects the FIR re-samplers used when
   * linear phase is enabled.
   * @param value the new number of samples that will be processed by each fft call.
   */
  void setFirFftBlockSize(uint32_t value)
  {
    oversampling32.setFirFftBlockSize(value);
    oversampling64.setFirFftBlockSize(value);
  }

  /**
   * Sets the antialiasing fir filter transition band. Only affects the behaviour of the object when linear phase is
   * enabled.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setFirTransitionBand(double transitionBand)
  {
    oversampling32.setFirTransitionBand(transitionBand);
    oversampling64.setFirTransitionBand(transitionBand);
  }

  /**
   * Sets whether the object shoul use the linear phase FIR re-samplers or the minimum-phase IIR re-samplers.
   * @param useLinearPhase true to enable linear phase, false to disable it.
   * */
  void setUseLinearPhase(bool useLinearPhase)
  {
    oversampling32.setUseLinearPhase(useLinearPhase);
    oversampling64.setUseLinearPhase(useLinearPhase);
  }

  /**
   * Sets the order of oversampling to be used. It must be less or equal to the maximum order set
   * @value the order to set
   * @return true if the order was set correctly, false otherwise
   */
  void setOrder(uint32_t order)
  {
    oversampling32.setUseLinearPhase(order);
    oversampling64.setUseLinearPhase(order);
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    oversampling32.reset();
    oversampling64.reset();
  }

  /**
   * @return the number of input samples to the up-sampling call needed before a first output sample is
   * produced by the up-sampling call.
   */
  uint32_t getUpSamplingLatency()
  {
    return oversampling32.getUpSamplingLatency();
  }

  /**
   * @return the number of input samples to the down-sampling call needed before a first output sample is
   * produced by the down-sampling call.
   */
  uint32_t getDownSamplingLatency()
  {
    return oversampling32.getDownSamplingLatency();
  }

  /**
   * @return the number of input samples to the up-sampling call needed before a first output sample is
   * produced by the down-sampling call.
   */
  uint32_t getLatency()
  {
    return oversampling32.getLatency();
  }

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers.
   */
  uint32_t getMaxNumOutputSamples()
  {
    return oversampling32.getMaxNumOutputSamples();
  }

  /**
   * @return the oversampling order currently in use.
   */
  uint32_t getOversamplingOrder() const
  {
    return oversampling32.getOversamplingOrder();
  }

  /**
   * @return the oversampling rate currently in use.
   */
  uint32_t getOversamplingRate() const
  {
    return oversampling32.getOversamplingRate();
  }

  /**
   * Sets the type of output for the up-sampling.
   * @param bufferType the type of output for the up-sampling.
   */
  void setUpSampledOutputBufferType(BufferType bufferType)
  {
    oversampling32.setUpSampledOutputBufferType(bufferType);
    oversampling64.setUpSampledOutputBufferType(bufferType);
  }

  /**
   * Sets the type of output for the down-sampling.
   * @param bufferType the type of output for the down-sampling.
   */
  void setDownSampledOutputBufferType(BufferType bufferType)
  {
    oversampling32.setDownSampledOutputBufferType(bufferType);
    oversampling64.setDownSampledOutputBufferType(bufferType);
  }

  /**
   * Sets the type of input for the down-sampling.
   * @param bufferType the type of input for the down-sampling.
   */
  void setDownSampledInputBufferType(BufferType bufferType)
  {
    oversampling32.setDownSampledInputBufferType(bufferType);
    oversampling64.setDownSampledInputBufferType(bufferType);
  }

  /**
   * Up-samples the input to a buffer owned by the object.
   * @param input pointer to the input buffers
   * @param numSamples the number of samples in each channel of the input buffer
   */
  template<class Float>
  uint32_t upSample(Float* const* input, uint32_t numSamples)
  {
    return get<Float>().upSample(input, numSamples);
  }

  /**
   * Up-samples the input to a buffer owned by the object.
   * @param input pointer to the input buffers
   */
  template<class Float>
  uint32_t upSample(InterleavedBuffer<Float> const& input)
  {
    return get<Float>().upSample(input);
  }

  /**
   * @return an interleaved buffer that holds the output of the up-sampling.
   */
  template<class Float>
  InterleavedBuffer<Float>& getUpSampleOutputInterleaved()
  {
    return get<Float>().getUpSampleOutputInterleaved();
  }

  /**
   * @return a buffer that holds the output of the up-sampling.
   */
  template<class Float>
  Buffer<Float>& getUpSampleOutput()
  {
    return get<Float>().getUpSampleOutput();
  }

  /**
   * @return a const interleaved buffer that holds the output of the up-sampling.
   */
  template<class Float>
  InterleavedBuffer<Float> const& getUpSampleOutputInterleaved() const
  {
    return get<Float>().getUpSampleOutputInterleaved();
  }

  /**
   * @return a const buffer that holds the output of the up-sampling.
   */
  template<class Float>
  Buffer<Float> const& getUpSampleOutput() const
  {
    return get<Float>().getUpSampleOutput();
  }

  /**
   * Down-samples the input.
   * @param input pointer to the input buffer.
   * @param numInputSamples the number of samples of each channel of the input
   * buffer.
   * @param output pointer to the memory in which to store the down-sampled data.
   * @param numOutputSamples the number of samples needed as output
   */
  template<class Float>
  void downSample(Float* const* input, uint32_t numInputSamples, Float** output, uint32_t numOutputSamples)
  {
    get<Float>().downSample(input, numInputSamples, output, numOutputSamples);
  }

  /**
   * Down-samples the input.
   * @param input an interleaved buffer holding the input.
   * @param output pointer to the memory in which to store the down-sampled data.
   * @param numOutputSamples the number of samples needed as output
   */
  template<class Float>
  void downSample(InterleavedBuffer<Float> const& input, Float** output, uint32_t numOutputSamples)
  {
    get<Float>().downSample(input, output, numOutputSamples);
  }

  /**
   * Down-samples the input to an InterleavedBuffer owned by the object.
   * @param input pointer to the input buffer.
   * @param numInputSamples the number of samples of each channel of the input
   * buffer.
   * @param numOutputSamples the number of samples needed as output
   */
  template<class Float>
  void downSample(Float* const* input, uint32_t numInputSamples, uint32_t numOutputSamples)
  {
    get<Float>().downSample(input, numInputSamples, numOutputSamples);
  }

  /**
   * Down-samples the input to an InterleavedBuffer owned by the object.
   * @param input an interleaved buffer holding the input.
   * @param numOutputSamples the number of samples needed as output
   */
  template<class Float>
  void downSample(InterleavedBuffer<Float> const& input, uint32_t numOutputSamples)
  {
    get<Float>().downSample(input, numOutputSamples);
  }

  /**
   * @return an interleaved buffer that holds the output of the down-sampling.
   */
  template<class Float>
  InterleavedBuffer<Float>& getDownSampleOutputInterleaved()
  {
    return get<Float>().getDownSampleOutputInterleaved();
  }

  /**
   * @return a const interleaved buffer that holds the output of the down-sampling.
   */
  template<class Float>
  InterleavedBuffer<Float> const& getDownSampleOutputInterleaved() const
  {
    return get<Float>().getDownSampleOutputInterleaved();
  }

private:
  template<class Float>
  oversimple::TOversampling<Float>& get();

  template<class Float>
  oversimple::TOversampling<Float> const& get() const;

  oversimple::TOversampling<float> oversampling32;
  oversimple::TOversampling<double> oversampling64;
};

template<>
inline oversimple::TOversampling<float>& Oversampling::get()
{
  return oversampling32;
}

template<>
inline oversimple::TOversampling<double>& Oversampling::get()
{
  return oversampling64;
}

template<>
inline oversimple::TOversampling<float> const& Oversampling::get() const
{
  return oversampling32;
}

template<>
inline oversimple::TOversampling<double> const& Oversampling::get() const
{
  return oversampling64;
}

} // namespace oversimple