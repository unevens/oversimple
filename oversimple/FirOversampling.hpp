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

namespace oversimple::fir::detail {

/**
 * Abstract class for FIR reSamplers, implementing getters, setters, filters,
 * and buffer management.
 */

class ReSamplerBase
{
public:
  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value);

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const
  {
    return numChannels;
  }

  /**
   * Sets the overampling rate.
   * @param value the new overampling factor.
   */
  virtual void setRate(double value);

  /**
   * @return the oversampling rate.
   */
  virtual double getRate() const
  {
    return oversamplingRate;
  }

  /**
   * Sets the number of samples that will be processed together.
   * @param value the new number of samples that will be processed together.
   */
  void setMaxSamplesPerBlock(int value);

  /**
   * @return the number of samples that will be processed together.
   */
  int getMaxSamplesPerBlock() const
  {
    return maxSamplesPerBlock;
  }

  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setTransitionBand(int value);

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
  int getNumSamplesBeforeOutputStarts();

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers. If prepareBuffers has not been called, then no
   * more samples than maxSamplesPerBlock should be passed to processBlock.
   */
  int getMaxNumOutputSamples() const
  {
    return maxOutputLength;
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  virtual void reset();

  /**
   * Prepare the reSampler to be able to process up to numSamples samples with
   * each processBlock call.
   * @param numSamples expected number of samples to be processed on each call
   * to processBlock.
   */
  virtual void prepareBuffers(int numSamples);

protected:
  double oversamplingRate;
  int numChannels;
  int maxSamplesPerBlock;
  double transitionBand;
  std::vector<std::unique_ptr<r8b::CDSPResampler24>> reSamplers;
  int maxOutputLength;
  int maxInputLength;

  virtual void setup();

  ReSamplerBase(int numChannels, double transitionBand, int maxSamplesPerBlock, double oversamplingRate);
};

/**
 * ReSampler using a FIR antialiasing filter. It will output every sample
 * produced, without buffering. As such, is better used for up-sampling. Only
 * works with double precision input/output.
 * @see TUnbufferedReSampler for a template that can work with single
 * precision.
 */
class UnbufferedReSampler : public ReSamplerBase
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit UnbufferedReSampler(int numChannels,
                               double transitionBand = 2.0,
                               int maxSamplesPerBlock = 1024,
                               double oversamplingRate = 1.0);

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @param output reference to a ScalarBuffer that holds the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(double* const* input, int numChannels, int numSamples);

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @param output reference to a ScalarBuffer that holds the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<double> const& input);

  ScalarBuffer<double>& getInternalOutput()
  {
    return output;
  }

  ScalarBuffer<double> const& getInternalOutput() const
  {
    return output;
  }

protected:
  ScalarBuffer<double> output;

private:
  void setup() override;
};

/**
 * UpSampler using a FIR antialiasing filter. Actually an alias for
 * UnbufferedReSampler. Only works with double precision input/output.
 * @see TUpSampler for a template that can work with single
 * precision.
 */
using UpSampler = UnbufferedReSampler;

/**
 * ReSampler using a FIR antialiasing filter. Its processing method takes a
 * number of requested samples, and will output either output that much samples,
 * or no samples at all. It uses a buffer to store the samples produced but not
 * works with double precision input/output.
 * @see TUnbufferedReampler for a template that can work with single
 * precision.
 */
class BufferedReSampler : public ReSamplerBase
{
  ScalarBuffer<double> buffer;
  int bufferCounter = 0;
  int maxRequiredOutputLength;

protected:
  void setup() override;

public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit BufferedReSampler(int numChannels,
                             double transitionBand = 2.0,
                             int maxSamplesPerBlock = 1024,
                             double oversamplingRate = 1.0);

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(double* const* input, int numSamples, double** output, int numOutputChannels, int requiredSamples);

  /**
   * Resamples a multi channel input buffer.
   * @param input a ScalarBuffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<double> const& input, double** output, int numOutputChannels, int requiredSamples);

  /**
   * Resamples a multi channel input buffer.
   * @param input a ScalarBuffer that holds the input buffer.
   * @param output a ScalarBuffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<double> const& input, ScalarBuffer<double>& output, int requiredSamples);

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   */
  virtual void prepareBuffers(int numInputSamples, int requiredOutputSamples);

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset() override;
};

/**
 * DownSampler using a FIR antialiasing filter. Only works with double precision
 * input/output.
 * @see TDownSampler for a template that can work with single
 * precision.
 */
class DownSampler final : public BufferedReSampler
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit DownSampler(int numChannels,
                       double transitionBand = 2.0,
                       int maxSamplesPerBlock = 1024,
                       double oversamplingRate = 1.0)
    : BufferedReSampler(numChannels, transitionBand, maxSamplesPerBlock, 1.0 / oversamplingRate)
  {}

  /**
   * Sets the overampling rate.
   * @param value the new overampling rate.
   */
  void setRate(double value) override
  {
    oversamplingRate = 1.0 / value;
    setup();
  }

  /**
   * @return the oversampling rate.
   */
  double getRate() const override
  {
    return 1.0 / oversamplingRate;
  }
};

/**
 * Template version of UnbufferedReSampler.
 * @see UnbufferedReSampler
 */
template<typename Scalar>
class TUnbufferedReampler final : public UnbufferedReSampler
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit TUnbufferedReampler(int numChannels,
                               double transitionBand = 2.0,
                               int maxSamplesPerBlock = 1024,
                               double oversamplingRate = 1.0)
    : UnbufferedReSampler(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
  {}

  ScalarBuffer<Scalar>& getOutput()
  {
    return output;
  }

  ScalarBuffer<Scalar> const& getOutput() const
  {
    return output;
  }
};

template<>
class TUnbufferedReampler<float> final : public UnbufferedReSampler
{
  ScalarBuffer<double> floatToDoubleBuffer;
  ScalarBuffer<float> doubleToFloatBuffer;

public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit TUnbufferedReampler(int numChannels,
                               double transitionBand = 2.0,
                               int maxSamplesPerBlock = 1024,
                               double oversamplingRate = 1.0)
    : UnbufferedReSampler(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
    , floatToDoubleBuffer(numChannels, maxSamplesPerBlock)
    , doubleToFloatBuffer(numChannels, (int)std::ceil(maxSamplesPerBlock * oversamplingRate))
  {}

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannelsToProcess number of channels to process
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @param output ScalarBuffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(float* const* input, int numChannelsToProcess, int numSamples)
  {
    assert(numChannelsToProcess <= numChannels);
    floatToDoubleBuffer.setNumSamples(numSamples);
    doubleToFloatBuffer.setNumSamples((int)std::ceil(numSamples * oversamplingRate));
    for (int c = 0; c < numChannelsToProcess; ++c) {
      for (int i = 0; i < numSamples; ++i) {
        floatToDoubleBuffer[c][i] = (double)input[c][i];
      }
    }
    int samples = UnbufferedReSampler::processBlock(floatToDoubleBuffer.get(), numChannelsToProcess, numSamples);
    copyScalarBuffer(output, doubleToFloatBuffer);
    return samples;
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @param output ScalarBuffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<float> const& input)
  {
    return processBlock(input.get(), input.getNumChannels(), input.getNumSamples());
  }

  void prepareBuffers(int numSamples) override
  {
    ReSamplerBase::prepareBuffers(numSamples);
    floatToDoubleBuffer.setNumSamples(numSamples);
    doubleToFloatBuffer.setNumSamples((int)std::ceil(numSamples * oversamplingRate));
  }

  ScalarBuffer<float>& getOutput()
  {
    return doubleToFloatBuffer;
  }

  ScalarBuffer<float> const& getOutput() const
  {
    return doubleToFloatBuffer;
  }
};

/**
 * Template version of UpSampler.
 * @see UpSampler
 */
template<typename Scalar>
using TUpSampler = TUnbufferedReampler<Scalar>;

/**
 * Template version of BufferedReSampler.
 * @see BufferedReSampler
 */
template<typename Scalar>
class TBufferedReSampler : public BufferedReSampler
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit TBufferedReSampler(int numChannels,
                              double transitionBand = 2.0,
                              int maxSamplesPerBlock = 1024,
                              double oversamplingRate = 1.0)
    : BufferedReSampler(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
  {}
};

template<>
class TBufferedReSampler<float> : public BufferedReSampler
{
  ScalarBuffer<double> floatToDoubleBuffer;
  ScalarBuffer<double> doubleToFloatBuffer;

public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit TBufferedReSampler(int numChannels,
                              double transitionBand = 2.0,
                              int maxSamplesPerBlock = 1024,
                              double oversamplingRate = 1.0)
    : BufferedReSampler(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
    , floatToDoubleBuffer(numChannels, maxSamplesPerBlock)
    , doubleToFloatBuffer(numChannels, (int)std::ceil(maxSamplesPerBlock * oversamplingRate))
  {}

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numChannelsToProcess number of channels to process
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(float* const* input, int numSamples, float** output, int numChannelsToProcess, int requiredSamples)
  {
    assert(numChannelsToProcess <= numChannels);

    floatToDoubleBuffer.setNumSamples(numSamples);
    doubleToFloatBuffer.setNumSamples(requiredSamples);
    for (int c = 0; c < numChannelsToProcess; ++c) {
      std::copy(&input[c][0], &input[c][0] + numSamples, &floatToDoubleBuffer[c][0]);
    }
    BufferedReSampler::processBlock(
      floatToDoubleBuffer.get(), numSamples, doubleToFloatBuffer.get(), numChannelsToProcess, requiredSamples);
    for (int c = 0; c < numChannels; ++c) {
      for (int i = 0; i < requiredSamples; ++i) {
        output[c][i] = (float)doubleToFloatBuffer[c][i];
      }
    }
  }

  /**
   * Resamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<float> const& input, float** output, int numChannelsToProcess, int requiredSamples)
  {
    copyScalarBuffer(input, floatToDoubleBuffer);
    doubleToFloatBuffer.setNumSamples(requiredSamples);
    BufferedReSampler::processBlock(floatToDoubleBuffer.get(),
                                    floatToDoubleBuffer.getNumSamples(),
                                    doubleToFloatBuffer.get(),
                                    numChannelsToProcess,
                                    requiredSamples);
    for (int c = 0; c < numChannelsToProcess; ++c) {
      for (int i = 0; i < requiredSamples; ++i) {
        output[c][i] = (float)doubleToFloatBuffer[c][i];
      }
    }
  }

  /**
   * Resamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output ScalarBuffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<float> const& input, ScalarBuffer<float>& output, int requiredSamples)
  {
    processBlock(input, output.get(), output.getNumChannels(), requiredSamples);
  }

  void prepareBuffers(int numInputSamples, int requiredOutputSamples) override
  {
    BufferedReSampler::prepareBuffers(numInputSamples, requiredOutputSamples);
    floatToDoubleBuffer.setNumSamples(numInputSamples);
    doubleToFloatBuffer.setNumSamples(requiredOutputSamples);
  }
};

/**
 * Template version of DownSampler.
 * @see DownSampler
 */
template<typename Scalar>
class TDownSampler final : public TBufferedReSampler<Scalar>
{
public:
  /**
   * Constructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  explicit TDownSampler(int numChannels,
                        double transitionBand = 2.0,
                        int maxSamplesPerBlock = 1024,
                        double oversamplingRate = 1.0)
    : TBufferedReSampler<Scalar>(numChannels, transitionBand, maxSamplesPerBlock, 1.0 / oversamplingRate)
  {}

  /**
   * Sets the overampling factor.
   * @param value the new overampling factor.
   */
  void setRate(double value) override
  {
    this->oversamplingRate = 1.0 / value;
    this->setup();
  }

  /**
   * @return the oversampling factor.
   */
  double getRate() const override
  {
    return 1.0 / this->oversamplingRate;
  }

  void prepareBuffers(int numSamplesWithoutOersampling) override
  {
    auto const numOversampledSamples = numSamplesWithoutOersampling * this->oversamplingRate;
    TBufferedReSampler<Scalar>::prepareBuffers(numOversampledSamples, numSamplesWithoutOersampling);
  }
};

template<class ReSampler>
class TReSamplerPreAllocatedBase
{
public:
  explicit TReSamplerPreAllocatedBase(int maxOrder = 5,
                                      int numChannels = 2,
                                      double transitionBand = 2.0,
                                      int maxSamplesPerBlock = 1024)
    : numChannels{ numChannels }
    , transitionBand{ transitionBand }
    , maxSamplesPerBlock{ maxSamplesPerBlock }
  {
    setMaxOrder(maxOrder);
  }

  virtual ~TReSamplerPreAllocatedBase() = default;

  void setMaxOrder(int value)
  {
    reSamplers.resize(static_cast<std::size_t>(value));
    int instanceOrder = 0;
    for (auto& reSampler : reSamplers) {
      if (!reSampler) {
        auto const rate = static_cast<double>(1 << instanceOrder);
        reSampler = std::make_unique<ReSampler>(numChannels, transitionBand, maxSamplesPerBlock, rate);
        reSampler->prepareBuffers(maxInputSamples);
      }
      ++instanceOrder;
    }
  }

  void setOrder(int value)
  {
    assert(value >= 0 && value <= static_cast<int>(reSamplers.size()));
    order = value;
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value)
  {
    numChannels = value;
    for (auto& reSampler : reSamplers) {
      reSampler->setNumChannels(value);
    }
  }

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const
  {
    return numChannels;
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void prepareBuffers(int numInputSamples)
  {
    maxInputSamples = numInputSamples;
    for (auto& reSampler : reSamplers) {
      reSampler->prepareBuffers(numInputSamples);
    }
  }

  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setTransitionBand(int value)
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
   * Sets the number of samples that will be processed together.
   * @param value the new number of samples that will be processed together.
   */
  void setMaxSamplesPerBlock(int value)
  {
    maxSamplesPerBlock = value;
    for (auto& reSampler : reSamplers) {
      reSampler->setMaxSamplesPerBlock(value);
    }
  }

  /**
   * @return the number of samples that will be processed together.
   */
  int getMaxSamplesPerBlock() const
  {
    return maxSamplesPerBlock;
  }

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  int getNumSamplesBeforeOutputStarts()
  {
    return get().getNumSamplesBeforeOutputStarts();
  }

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers. If prepareBuffers has not been called, then no
   * more samples than maxSamplesPerBlock should be passed to processBlock.
   */
  int getMaxNumOutputSamples() const
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
  ReSampler& get()
  {
    return *reSamplers[order];
  }

  ReSampler const& get() const
  {
    return *reSamplers[order];
  }

private:
  std::vector<std::unique_ptr<ReSampler>> reSamplers;
  int numChannels = 2;
  int maxInputSamples = 256;
  int maxSamplesPerBlock = 1024;
  double transitionBand = 2.0;
  int order = 0;
};

template<typename Scalar>
class TUpSamplerPreAllocated final : public TReSamplerPreAllocatedBase<TUpSampler<Scalar>>
{
public:
  using TReSamplerPreAllocatedBase<TUpSampler<Scalar>>::TReSamplerPreAllocatedBase;

  ScalarBuffer<double>& getOutput()
  {
    return this->get().getOutput;
  }

  ScalarBuffer<double> const& getOutput() const
  {
    return this->get().getOutput;
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @return number of upsampled samples
   */
  int processBlock(double* const* input, int numChannels, int numSamples)
  {
    return this->get().processBlock(input, numChannels, numSamples);
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<double> const& input)
  {
    return this->get().processBlock(input);
  }
};

template<typename Scalar>
using TDownSamplerPreAllocated = TReSamplerPreAllocatedBase<TDownSampler<Scalar>>;

} // namespace oversimple::fir
