/*
Copyright 2019-2021 Dario Mambro

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
#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

#include "CDSPResampler.h"

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "avec/Avec.hpp"

namespace oversimple::fir {

/**
 * Base class for FIR ReSamplers, implementing getters, setters, filters and buffers management.
 */

class ReSamplerBase
{
public:
  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(uint32_t value);

  /**
   * @return the number of channels the processor is ready to work with.
   */
  uint32_t getNumChannels() const
  {
    return numChannels;
  }

  /**
   * Sets the number of samples that are processed by each fft call.
   * @param value the new number of samples that will be processed by each fft call.
   */
  void setFftSamplesPerBlock(uint32_t value);

  /**
   * @return the number of samples that are processed by each fft call.
   */
  uint32_t getFftSamplesPerBlock() const
  {
    return fftSamplesPerBlock;
  }

  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setTransitionBand(uint32_t value);

  /**
   * @return value the antialiasing filter transition band, in percentage of the
   * sample rate.
   */
  double getTransitionBand() const
  {
    return transitionBand;
  }

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  uint32_t getNumSamplesBeforeOutputStarts();

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers. If prepareBuffers has not been called, then no
   * more samples than fftSamplesPerBlock should be passed to processBlock.
   */
  uint32_t getMaxNumOutputSamples() const
  {
    return maxOutputLength;
  }

  virtual ~ReSamplerBase() = default;

protected:
  ReSamplerBase(uint32_t numChannels, double transitionBand, uint32_t fftSamplesPerBlock, double oversamplingRate);

  virtual void setup();

  void prepareBuffersBase(uint32_t numSamples);

  void resetBase();

  double oversamplingRate = 1.0;
  uint32_t numChannels = 2;
  uint32_t fftSamplesPerBlock = 512;
  double transitionBand = 2.0;
  std::vector<std::unique_ptr<r8b::CDSPResampler24>> reSamplers;
  uint32_t maxOutputLength = 0;
  uint32_t maxInputLength = 256;
};

/**
 * UpSampler using a FIR antialiasing filter. It will output every sample
 * produced, without buffering.
 * @see TUnbufferedReSampler for a template that can work with single
 * precision.
 */
class UpSampler : public ReSamplerBase
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   * @param oversamplingRate the oversampling factor
   */
  explicit UpSampler(uint32_t numChannels,
                     double transitionBand = 2.0,
                     uint32_t fftSamplesPerBlock = 256,
                     double oversamplingRate = 1.0);

  /**
   * Up-samples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @return number of up-sampled samples
   */
  uint32_t processBlock(double* const* input, uint32_t numSamples);

  /**
   * Up-samples a multi channel input buffer.
   * @param input Buffer that holds the input buffer.
   * @return number of upsampled samples
   */
  uint32_t processBlock(Buffer<double> const& input);

  Buffer<double>& getOutputBuffer()
  {
    return output;
  }

  Buffer<double> const& getOutputBuffer() const
  {
    return output;
  }

  /**
   * Prepare the reSampler to be able to process up to numSamples samples with
   * each processBlock call.
   * @param numSamples expected number of samples to be processed on each call
   * to processBlock.
   * @param fftBlockSize the new number of samples that will be processed by each fft call.
   */
  virtual void prepareBuffersAndSetFftBlockSize(uint32_t numSamples, uint32_t fftBlockSize);

  /**
   * Prepare the reSampler to be able to process up to numSamples samples with
   * each processBlock call.
   * @param numSamples expected number of samples to be processed on each call
   * to processBlock.
   */
  virtual void prepareBuffers(uint32_t numSamples);

  /**
   * Sets the oversampling rate.
   * @param value the new oversampling rate.
   */
  void setRate(double value)
  {
    oversamplingRate = value;
    setup();
  }

  /**
   * @return the oversampling rate.
   */
  double getRate() const
  {
    return oversamplingRate;
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    resetBase();
  }

protected:
  void setup() override;

  Buffer<double> output;
};

/**
 * DownSampler using a FIR antialiasing filter. Its processing method takes a
 * number of requested samples, and will output either output that much samples,
 * or no samples at all. It uses a buffer to store the samples produced.
 * @see TUnbufferedReSampler for a template that can work with single
 * precision.
 */
class DownSampler : public ReSamplerBase
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   * @param oversamplingRate the oversampling factor
   */
  explicit DownSampler(uint32_t numChannels,
                       double transitionBand = 2.0,
                       uint32_t fftSamplesPerBlock = 256,
                       double oversamplingRate = 1.0);

  /**
   * Down-samples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(double* const* input, uint32_t numSamples, double** output, uint32_t requiredSamples);

  /**
   * Down-samples a multi channel input buffer.
   * @param input a Buffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Buffer<double> const& input, double** output, uint32_t requiredSamples);

  /**
   * Down-samples a multi channel input buffer.
   * @param input a Buffer that holds the input buffer.
   * @param output a Buffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Buffer<double> const& input, Buffer<double>& output, uint32_t requiredSamples);

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   */
  virtual void prepareBuffers(uint32_t numInputSamples, uint32_t requiredOutputSamples);

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples. Also sets the fft block size.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   * @param fftBlockSize the new number of samples that will be processed by each fft call.
   */
  virtual void prepareBuffersAndSetFftBlockSize(uint32_t numInputSamples,
                                                uint32_t requiredOutputSamples,
                                                uint32_t fftBlockSize);

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    resetBase();
    bufferCounter = 0;
  }

  /**
   * Sets the oversampling rate.
   * @param value the new oversampling rate.
   */
  void setRate(double value)
  {
    oversamplingRate = 1.0 / value;
    setup();
  }

  /**
   * @return the oversampling rate.
   */
  double getRate() const
  {
    return 1.0 / oversamplingRate;
  }

private:
  Buffer<double> buffer;
  int bufferCounter = 0;
  uint32_t maxRequiredOutputLength;

protected:
  void setup() override;
  void updateBuffer(uint32_t requiredOutputSamples);
};

/**
 * Template version of UnbufferedReSampler.
 * @see UnbufferedReSampler
 */
template<typename Float>
class TUpSampler final : public UpSampler
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   * @param oversamplingRate the oversampling factor
   */
  explicit TUpSampler(uint32_t numChannels,
                      double transitionBand = 2.0,
                      uint32_t fftSamplesPerBlock = 256,
                      double oversamplingRate = 1.0)
    : UpSampler(numChannels, transitionBand, fftSamplesPerBlock, oversamplingRate)
  {}

  Buffer<Float>& getOutput()
  {
    return output;
  }

  Buffer<Float> const& getOutput() const
  {
    return output;
  }
};

template<>
class TUpSampler<float> final : public UpSampler
{
  Buffer<double> floatToDoubleBuffer;
  Buffer<float> doubleToFloatBuffer;

public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   * @param oversamplingRate the oversampling factor
   */
  explicit TUpSampler(uint32_t numChannels,
                      double transitionBand = 2.0,
                      uint32_t fftSamplesPerBlock = 256,
                      double oversamplingRate = 1.0)
    : UpSampler(numChannels, transitionBand, fftSamplesPerBlock, oversamplingRate)
    , floatToDoubleBuffer(numChannels, fftSamplesPerBlock)
    , doubleToFloatBuffer(numChannels, (uint32_t)std::ceil(fftSamplesPerBlock * oversamplingRate))
  {}

  /**
   * Up-samples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @return number of upsampled samples
   */
  uint32_t processBlock(float* const* input, uint32_t numSamples)
  {
    auto const numUpSampledSamples = (uint32_t)std::ceil(numSamples * oversamplingRate);
    assert(floatToDoubleBuffer.getCapacity() >= numSamples);
    assert(doubleToFloatBuffer.getCapacity() >= numUpSampledSamples);
    floatToDoubleBuffer.setNumSamples(numSamples);
    doubleToFloatBuffer.setNumSamples(numUpSampledSamples);
    for (uint32_t c = 0; c < numChannels; ++c) {
      for (uint32_t i = 0; i < numSamples; ++i) {
        floatToDoubleBuffer[c][i] = (double)input[c][i];
      }
    }
    auto const samples = UpSampler::processBlock(floatToDoubleBuffer.get(), numSamples);
    copyBuffer(output, doubleToFloatBuffer);
    return samples;
  }

  /**
   * Up-samples a multi channel input buffer.
   * @param input Buffer that holds the input buffer.
   * @return number of upsampled samples
   */
  uint32_t processBlock(Buffer<float> const& input)
  {
    return processBlock(input.get(), input.getNumSamples());
  }

  /**
   * Allocates resources to process up to numInputSamples input samples.
   * @param numInputSamples the expected maximum number input samples
   * @param fftBlockSize the new number of samples that will be processed by each fft call.
   */
  void prepareBuffersAndSetFftBlockSize(uint32_t numSamples, uint32_t fftBlockSize) override
  {
    UpSampler::prepareBuffersAndSetFftBlockSize(numSamples, fftBlockSize);
    updateBuffers(numSamples);
  }

  /**
   * Allocates resources to process up to numInputSamples input samples.
   * @param numInputSamples the expected maximum number input samples
   */
  void prepareBuffers(uint32_t numSamples) override
  {
    UpSampler::prepareBuffers(numSamples);
    updateBuffers(numSamples);
  }

  Buffer<float>& getOutput()
  {
    return doubleToFloatBuffer;
  }

  Buffer<float> const& getOutput() const
  {
    return doubleToFloatBuffer;
  }

private:
  void setup() override
  {
    floatToDoubleBuffer.setNumChannels(this->numChannels);
    doubleToFloatBuffer.setNumChannels(this->numChannels);
    UpSampler::setup();
  }

  void updateBuffers(uint32_t numSamples)
  {
    floatToDoubleBuffer.setNumSamples(numSamples);
    doubleToFloatBuffer.setNumSamples((uint32_t)std::ceil(numSamples * oversamplingRate));
  }
};

/**
 * Template version of BufferedReSampler.
 * @see BufferedReSampler
 */
template<typename Float>
class TDownSampler : public DownSampler
{
public:
  using DownSampler::DownSampler;
};

template<>
class TDownSampler<float> : public DownSampler
{
  Buffer<double> floatToDoubleBuffer;
  Buffer<double> doubleToFloatBuffer;

public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   * @param oversamplingRate the oversampling factor
   */
  explicit TDownSampler(uint32_t numChannels,
                        double transitionBand = 2.0,
                        uint32_t fftSamplesPerBlock = 256,
                        double oversamplingRate_ = 1.0)
    : DownSampler(numChannels, transitionBand, fftSamplesPerBlock, oversamplingRate_)
    , floatToDoubleBuffer(numChannels, fftSamplesPerBlock)
    , doubleToFloatBuffer(numChannels, (uint32_t)std::ceil(fftSamplesPerBlock * oversamplingRate))
  {}

  /**
   * Down-samples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(float* const* input, uint32_t numSamples, float** output, uint32_t requiredSamples)
  {
    assert(floatToDoubleBuffer.getCapacity() >= numSamples);
    assert(doubleToFloatBuffer.getCapacity() >= requiredSamples);
    floatToDoubleBuffer.setNumSamples(numSamples);
    doubleToFloatBuffer.setNumSamples(requiredSamples);
    if (numSamples > 0) {
      for (uint32_t c = 0; c < numChannels; ++c) {
        std::copy(&input[c][0], &input[c][0] + numSamples, &floatToDoubleBuffer[c][0]);
      }
    }
    DownSampler::processBlock(floatToDoubleBuffer.get(), numSamples, doubleToFloatBuffer.get(), requiredSamples);
    for (uint32_t c = 0; c < numChannels; ++c) {
      for (uint32_t i = 0; i < requiredSamples; ++i) {
        output[c][i] = (float)doubleToFloatBuffer[c][i];
      }
    }
  }

  /**
   * Down-samples a multi channel input buffer.
   * @param Buffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Buffer<float> const& input, float** output, uint32_t requiredSamples)
  {
    assert(floatToDoubleBuffer.getCapacity() >= input.getNumSamples());
    assert(doubleToFloatBuffer.getCapacity() >= requiredSamples);
    copyBuffer(input, floatToDoubleBuffer);
    doubleToFloatBuffer.setNumSamples(requiredSamples);
    DownSampler::processBlock(
      floatToDoubleBuffer.get(), floatToDoubleBuffer.getNumSamples(), doubleToFloatBuffer.get(), requiredSamples);
    for (uint32_t c = 0; c < numChannels; ++c) {
      for (uint32_t i = 0; i < requiredSamples; ++i) {
        output[c][i] = (float)doubleToFloatBuffer[c][i];
      }
    }
  }

  /**
   * Down-samples a multi channel input buffer.
   * @param Buffer that holds the input buffer.
   * @param output Buffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Buffer<float> const& input, Buffer<float>& output, uint32_t requiredSamples)
  {
    processBlock(input, output.get(), requiredSamples);
  }

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   * @param fftBlockSize the new number of samples that will be processed by each fft call.
   */
  void prepareBuffersAndSetFftBlockSize(uint32_t numInputSamples,
                                        uint32_t requiredOutputSamples,
                                        uint32_t fftBlockSize) override
  {
    DownSampler::prepareBuffersAndSetFftBlockSize(numInputSamples, requiredOutputSamples, fftBlockSize);
    updateBuffers(numInputSamples, requiredOutputSamples);
  }

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   */
  void prepareBuffers(uint32_t numInputSamples, uint32_t requiredOutputSamples) override
  {
    DownSampler::prepareBuffers(numInputSamples, requiredOutputSamples);
    updateBuffers(numInputSamples, requiredOutputSamples);
  }

protected:
  void setup() override
  {
    floatToDoubleBuffer.setNumChannels(this->numChannels);
    doubleToFloatBuffer.setNumChannels(this->numChannels);
    DownSampler::setup();
  }

private:
  void updateBuffers(uint32_t numInputSamples, uint32_t requiredOutputSamples)
  {
    floatToDoubleBuffer.setNumSamples(numInputSamples);
    doubleToFloatBuffer.setNumSamples(requiredOutputSamples);
  }
};

template<class ReSampler>
class TReSamplerPreAllocatedBase
{
public:
  virtual ~TReSamplerPreAllocatedBase() = default;

  /**
   * Sets the order of oversampling to be used. It must be less or equal to the maximum order set
   * @value the order to set
   * @return true if the order was set correctly, false otherwise
   */
  bool setOrder(uint32_t value)
  {
    if (value >= 1 && value <= static_cast<uint32_t>(reSamplers.size())) {
      order = value;
      return true;
    }
    return false;
  }

  /**
   * @return the order of oversampling currently in use
   */
  uint32_t getOrder() const
  {
    return order;
  }

  /**
   * @return the oversampling rate currently in use
   */
  uint32_t getRate() const
  {
    return 1 << order;
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(uint32_t value)
  {
    numChannels = value;
    for (auto& reSampler : reSamplers) {
      reSampler->setNumChannels(value);
    }
  }

  /**
   * @return the number of channels the processor is ready to work with.
   */
  uint32_t getNumChannels() const
  {
    return numChannels;
  }

  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setTransitionBand(uint32_t value)
  {
    transitionBand = value;
    for (auto& reSampler : reSamplers) {
      reSampler->setTransitionBand(value);
    }
  }

  /**
   * @return value the antialiasing filter transition band, in percentage of the
   * sample rate.
   */
  double getTransitionBand() const
  {
    return transitionBand;
  }

  /**
   * Sets the number of samples that are processed by each fft call.
   * @param value the new number of samples that will be processed by each fft call.
   */
  void setFftSamplesPerBlock(uint32_t value)
  {
    fftSamplesPerBlock = value;
    for (auto& reSampler : reSamplers) {
      reSampler->setFftSamplesPerBlock(value);
    }
  }

  /**
   * @return the number of samples that will be processed together.
   */
  uint32_t getFftSamplesPerBlock() const
  {
    return fftSamplesPerBlock;
  }

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  uint32_t getNumSamplesBeforeOutputStarts()
  {
    return get().getNumSamplesBeforeOutputStarts();
  }

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers. If prepareBuffers has not been called, then no
   * more samples than fftSamplesPerBlock should be passed to processBlock.
   */
  uint32_t getMaxNumOutputSamples() const
  {
    return get().getMaxNumOutputSamples();
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    get().reset();
  }

protected:
  explicit TReSamplerPreAllocatedBase(uint32_t numChannels = 2,
                                      double transitionBand = 2.0,
                                      uint32_t fftSamplesPerBlock = 512)
    : numChannels{ numChannels }
    , transitionBand{ transitionBand }
    , fftSamplesPerBlock{ fftSamplesPerBlock }
  {}

  ReSampler& get()
  {
    return *reSamplers[order - 1];
  }

  ReSampler const& get() const
  {
    return *reSamplers[order - 1];
  }

  std::vector<std::unique_ptr<ReSampler>> reSamplers;
  uint32_t numChannels = 2;
  uint32_t maxInputSamples = 256;
  uint32_t fftSamplesPerBlock = 512;
  double transitionBand = 2.0;
  uint32_t order = 1;
};

template<typename Float>
class TUpSamplerPreAllocated final : public TReSamplerPreAllocatedBase<TUpSampler<Float>>
{
public:
  /**
   * Constructor.
   * @param maxOrder the maximum order of oversampling to allocate resources for
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   */
  explicit TUpSamplerPreAllocated(uint32_t maxOrder = 5,
                                  uint32_t numChannels = 2,
                                  double transitionBand = 2.0,
                                  uint32_t fftSamplesPerBlock = 512)
    : TReSamplerPreAllocatedBase<TUpSampler<Float>>(numChannels, transitionBand, fftSamplesPerBlock)
  {
    setMaxOrder(maxOrder);
  }

  /**
   * @return a Buffer holding the output of the processing
   */
  Buffer<Float>& getOutput()
  {
    return this->get().getOutput();
  }

  /**
   * @return a const Buffer holding the output of the processing
   */
  Buffer<Float> const& getOutput() const
  {
    return this->get().getOutput();
  }

  /**
   * Sets the maximum order of oversampling supported, and allocates the necessary resources
   * @param value the maximum order of oversampling
   */
  void setMaxOrder(uint32_t value)
  {
    this->reSamplers.resize(static_cast<std::size_t>(value));
    uint32_t instanceOrder = 1;
    for (auto& reSampler : this->reSamplers) {
      if (!reSampler) {
        auto const rate = static_cast<double>(1 << instanceOrder);
        reSampler =
          std::make_unique<TUpSampler<Float>>(this->numChannels, this->transitionBand, this->fftSamplesPerBlock, rate);
        reSampler->prepareBuffers(this->maxInputSamples);
      }
      ++instanceOrder;
    }
  }

  /**
   * Allocates resources to process up to numInputSamples input.
   * @param numInputSamples the expected maximum number input samples
   * @param fftBlockSize the new number of samples that will be processed by each fft call.
   */
  void prepareBuffersAndSetFftBlockSize(uint32_t numInputSamples, uint32_t fftBlockSize)
  {
    this->maxInputSamples = numInputSamples;
    this->fftSamplesPerBlock = fftBlockSize;
    for (auto& reSampler : this->reSamplers) {
      reSampler->prepareBuffersAndSetFftBlockSize(numInputSamples, fftBlockSize);
    }
  }

  /**
   * Allocates resources to process up to numInputSamples input.
   * @param numInputSamples the expected maximum number input samples
   */
  void prepareBuffers(uint32_t numInputSamples)
  {
    this->maxInputSamples = numInputSamples;
    for (auto& reSampler : this->reSamplers) {
      reSampler->prepareBuffers(numInputSamples);
    }
  }

  /**
   * Up-samples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @return number of upsampled samples
   */
  uint32_t processBlock(Float* const* input, uint32_t numSamples)
  {
    return this->get().processBlock(input, numSamples);
  }

  /**
   * Up-samples a multi channel input buffer.
   * @param input Buffer that holds the input buffer.
   * @return number of upsampled samples
   */
  uint32_t processBlock(Buffer<Float> const& input)
  {
    return this->get().processBlock(input);
  }
};

template<typename Float>
class TDownSamplerPreAllocated final : public TReSamplerPreAllocatedBase<TDownSampler<Float>>
{
public:
  /**
   * Constructor.
   * @param maxOrder the maximum order of oversampling to allocate resources for
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param fftSamplesPerBlock the number of samples that will be processed
   * by each fft call.
   */
  explicit TDownSamplerPreAllocated(uint32_t maxOrder = 5,
                                    uint32_t numChannels = 2,
                                    double transitionBand = 2.0,
                                    uint32_t fftSamplesPerBlock = 512)
    : TReSamplerPreAllocatedBase<TDownSampler<Float>>(numChannels, transitionBand, fftSamplesPerBlock)
  {
    setMaxOrder(maxOrder);
  }

  /**
   * Down-samples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Float* const* input, uint32_t numSamples, Float** output, uint32_t requiredSamples)
  {
    return this->get().processBlock(input, numSamples, output, requiredSamples);
  }

  /**
   * Down-samples a multi channel input buffer.
   * @param input a Buffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Buffer<Float> const& input, Float** output, uint32_t requiredSamples)
  {
    return this->get().processBlock(input, output, requiredSamples);
  }

  /**
   * Down-samples a multi channel input buffer.
   * @param input a Buffer that holds the input buffer.
   * @param output a Buffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Buffer<Float> const& input, Buffer<Float>& output, uint32_t requiredSamples)
  {
    return this->get().processBlock(input, output, requiredSamples);
  }

  /**
   * Sets the maximum order of oversampling supported, and allocates the necessary resources
   * @param value the maximum order of oversampling
   */
  void setMaxOrder(uint32_t value)
  {
    this->reSamplers.resize(static_cast<std::size_t>(value));
    uint32_t instanceOrder = 1;
    for (auto& reSampler : this->reSamplers) {
      if (!reSampler) {
        auto const rate = static_cast<double>(1 << instanceOrder);
        reSampler = std::make_unique<TDownSampler<Float>>(
          this->numChannels, this->transitionBand, this->fftSamplesPerBlock, rate);
        reSampler->prepareBuffers(this->maxInputSamples, maxRequiredOutputSamples);
      }
      ++instanceOrder;
    }
  }

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   * @param fftBlockSize the new number of samples that will be processed by each fft call.
   */
  void prepareBuffersAndSetFftBlockSize(uint32_t numInputSamples,
                                        uint32_t requiredOutputSamples_,
                                        uint32_t fftBlockSize)
  {
    maxRequiredOutputSamples = requiredOutputSamples_;
    this->maxInputSamples = numInputSamples;
    this->fftSamplesPerBlock = fftBlockSize;
    for (auto& reSampler : this->reSamplers) {
      reSampler->prepareBuffersAndSetFftBlockSize(numInputSamples, maxRequiredOutputSamples, fftBlockSize);
    }
  }

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   */
  void prepareBuffers(uint32_t numInputSamples, uint32_t requiredOutputSamples_)
  {
    maxRequiredOutputSamples = requiredOutputSamples_;
    this->maxInputSamples = numInputSamples;
    for (auto& reSampler : this->reSamplers) {
      reSampler->prepareBuffers(numInputSamples, maxRequiredOutputSamples);
    }
  }

private:
  uint32_t maxRequiredOutputSamples = 256;
};

} // namespace oversimple::fir
