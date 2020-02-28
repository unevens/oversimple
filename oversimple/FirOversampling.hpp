/*
Copyright 2019-2020 Dario Mambro

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
#include "CDSPResampler.h"
#include "avec/Avec.hpp"

namespace oversimple {

/**
 * Abstract class for FIR resamplers, implementing getters, setters, filters,
 * and buffer management.
 */

class FirResamplerBase
{
protected:
  double oversamplingRate;
  int numChannels;
  int maxSamplesPerBlock;
  double transitionBand;
  std::vector<std::unique_ptr<r8b::CDSPResampler24>> resamplers;
  int maxOutputLength;
  int maxInputLength;

  virtual void setup();
  FirResamplerBase(int numChannels,
                   double transitionBand,
                   int maxSamplesPerBlock,
                   double oversamplingRate);

public:
  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value);

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const { return numChannels; }

  /**
   * Sets the overampling rate.
   * @param value the new overampling factor.
   */
  virtual void setRate(double value);

  /**
   * @return the oversampling rate.
   */
  virtual double getRate() const { return oversamplingRate; }

  /**
   * Sets the number of samples that will be processed together.
   * @param value the new number of samples that will be processed together.
   */
  void setMaxSamplesPerBlock(int value);

  /**
   * @return the number of samples that will be processed together.
   */
  int getMaxSamplesPerBlock() const { return maxSamplesPerBlock; }

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
  double getTransitionBand() const { return transitionBand; }

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
  int getMaxNumOutputSamples() { return maxOutputLength; }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  virtual void reset();

  /**
   * Prepare the resampler to be able to process up to numSamples samples with
   * each processBlock call.
   * @param numSamples expected number of samples to be processed on each call
   * to processBlock.
   */
  void prepareBuffers(int numSamples);
};

/**
 * Resampler using a FIR antialiasing filter. It will output every sample
 * produced, without buffering. As such, is better used for upsampling. Only
 * works with double precision input/output.
 * @see TFirUnbufferedReampler for a template that can work with single
 * precision.
 */
class FirUnbufferedResampler : public FirResamplerBase
{
  void setup() override;

public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  FirUnbufferedResampler(int numChannels,
                         double transitionBand = 2.0,
                         int maxSamplesPerBlock = 1024,
                         double oversamplingRate = 1.0);

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(double* const* input,
                   int numChannels,
                   int numSamples,
                   ScalarBuffer<double>& output);

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<double> const& input,
                   ScalarBuffer<double>& output);
};

/**
 * Upsampler using a FIR antialiasing filter. Actually an alias for
 * FirUnbufferedResampler. Only works with double precision input/output.
 * @see TFirUpsampler for a template that can work with single
 * precision.
 */
using FirUpsampler = FirUnbufferedResampler;

/**
 * Resampler using a FIR antialiasing filter. Its processing method takes a
 * number of requested samples, and will output either output that much samples,
 * or no samples at all. It uses a buffer to store the samples produced but not
 * works with double precision input/output.
 * @see TFirUnbufferedReampler for a template that can work with single
 * precision.
 */
class FirBufferedResampler : public FirResamplerBase
{
  ScalarBuffer<double> buffer;
  int bufferCounter = 0;
  int maxRequiredOutputLength;

protected:
  void setup() override;

public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  FirBufferedResampler(int numChannels,
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
  void processBlock(double* const* input,
                    int numSamples,
                    double** output,
                    int numOutputChannels,
                    int requiredSamples);

  /**
   * Resamples a multi channel input buffer.
   * @param input a ScalarBuffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<double> const& input,
                    double** output,
                    int numOutputChannels,
                    int requiredSamples);

  /**
   * Resamples a multi channel input buffer.
   * @param input a ScalarBuffer that holds the input buffer.
   * @param output a ScalarBufffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<double> const& input,
                    ScalarBuffer<double>& output,
                    int requiredSamples);

  /**
   * Allocates resources to process up to numInputSamples input samples and
   * produce requiredOutputSamples output samples.
   * @param numInputSamples the expected maximum number input samples
   * @param requiredOutputSamples the required number of output samples
   */
  void prepareBuffers(int numInputSamples, int requiredOutputSamples);

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset() override;
};

/**
 * Downsampler using a FIR antialiasing filter. Only works with double precision
 * input/output.
 * @see TFirDownsampler for a template that can work with single
 * precision.
 */
class FirDownsampler : public FirBufferedResampler
{
public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  FirDownsampler(int numChannels,
                 double transitionBand = 2.0,
                 int maxSamplesPerBlock = 1024,
                 double oversamplingRate = 1.0)
    : FirBufferedResampler(numChannels,
                           transitionBand,
                           maxSamplesPerBlock,
                           1.0 / oversamplingRate)
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
  double getRate() const override { return 1.0 / oversamplingRate; }
};

/**
 * Template version of FirUnbufferedResampler.
 * @see FirUnbufferedResampler
 */
template<typename Scalar>
class TFirUnbufferedReampler final : public FirUnbufferedResampler
{
public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  TFirUnbufferedReampler(int numChannels,
                         double transitionBand = 2.0,
                         int maxSamplesPerBlock = 1024,
                         double oversamplingRate = 1.0)
    : FirUnbufferedResampler(numChannels,
                             transitionBand,
                             maxSamplesPerBlock,
                             oversamplingRate)
  {}
};

template<>
class TFirUnbufferedReampler<float> final : public FirUnbufferedResampler
{
  ScalarBuffer<double> floatToDoubleBuffer;
  ScalarBuffer<double> doubleToFloatBuffer;

public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  TFirUnbufferedReampler(int numChannels,
                         double transitionBand = 2.0,
                         int maxSamplesPerBlock = 1024,
                         double oversamplingRate = 1.0)
    : FirUnbufferedResampler(numChannels,
                             transitionBand,
                             maxSamplesPerBlock,
                             oversamplingRate)
  {}

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(float* const* input,
                   int numChannels,
                   int numSamples,
                   ScalarBuffer<float>& output)
  {
    floatToDoubleBuffer.setNumChannelsAndSamples(numChannels, numSamples);
    doubleToFloatBuffer.setNumChannelsAndSamples(
      numChannels, std::ceil(numSamples * oversamplingRate));
    for (int c = 0; c < numChannels; ++c) {
      for (int i = 0; i < numSamples; ++i) {
        floatToDoubleBuffer[c][i] = (double)input[c][i];
      }
    }
    int samples = FirUnbufferedResampler::processBlock(floatToDoubleBuffer,
                                                       doubleToFloatBuffer);
    copyScalarBuffer(doubleToFloatBuffer, output);
    return samples;
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<float> const& input,
                   ScalarBuffer<float>& output)
  {
    return processBlock(
      input.get(), input.getNumChannels(), input.getNumSamples(), output);
  }
};

/**
 * Template version of FirUpsampler.
 * @see FirUpsampler
 */
template<typename Scalar>
using TFirUpsampler = TFirUnbufferedReampler<Scalar>;

/**
 * Template version of FirBufferedResampler.
 * @see FirBufferedResampler
 */
template<typename Scalar>
class TFirBufferedResampler : public FirBufferedResampler
{
public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  TFirBufferedResampler(int numChannels,
                        double transitionBand = 2.0,
                        int maxSamplesPerBlock = 1024,
                        double oversamplingRate = 1.0)
    : FirBufferedResampler(numChannels,
                           transitionBand,
                           maxSamplesPerBlock,
                           oversamplingRate)
  {}
};

template<>
class TFirBufferedResampler<float> : public FirBufferedResampler
{
  ScalarBuffer<double> floatToDoubleBuffer;
  ScalarBuffer<double> doubleToFloatBuffer;

public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  TFirBufferedResampler(int numChannels,
                        double transitionBand = 2.0,
                        int maxSamplesPerBlock = 1024,
                        double oversamplingRate = 1.0)
    : FirBufferedResampler(numChannels,
                           transitionBand,
                           maxSamplesPerBlock,
                           oversamplingRate)
  {}

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(float* const* input,
                    int numSamples,
                    float** output,
                    int numOutputChannels,
                    int requiredSamples)
  {
    floatToDoubleBuffer.setNumChannelsAndSamples(numChannels, numSamples);
    for (int c = 0; c < numChannels; ++c) {
      std::copy(
        &input[c][0], &input[c][0] + numSamples, &floatToDoubleBuffer[c][0]);
    }
    doubleToFloatBuffer.setNumChannelsAndSamples(numChannels, requiredSamples);
    FirBufferedResampler::processBlock(
      floatToDoubleBuffer, doubleToFloatBuffer, requiredSamples);
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
  void processBlock(ScalarBuffer<float> const& input,
                    float** output,
                    int numChannels,
                    int requiredSamples)
  {
    copyScalarBuffer(input, floatToDoubleBuffer);
    doubleToFloatBuffer.setNumChannelsAndSamples(numChannels, requiredSamples);
    FirBufferedResampler::processBlock(
      floatToDoubleBuffer, doubleToFloatBuffer, requiredSamples);
    for (int c = 0; c < numChannels; ++c) {
      for (int i = 0; i < requiredSamples; ++i) {
        output[c][i] = (float)doubleToFloatBuffer[c][i];
      }
    }
  }

  /**
   * Resamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<float> const& input,
                    ScalarBuffer<float>& output,
                    int requiredSamples)
  {
    processBlock(input, output.get(), output.getNumChannels(), requiredSamples);
  }
};

/**
 * Template version of FirDownsampler.
 * @see FirDownsampler
 */
template<typename Scalar>
class TFirDownsampler final : public TFirBufferedResampler<Scalar>
{
public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingRate the oversampling factor
   */
  TFirDownsampler(int numChannels,
                  double transitionBand = 2.0,
                  int maxSamplesPerBlock = 1024,
                  double oversamplingRate = 1.0)
    : TFirBufferedResampler<Scalar>(numChannels,
                                    transitionBand,
                                    maxSamplesPerBlock,
                                    1.0 / oversamplingRate)
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
  double getRate() const override { return 1.0 / this->oversamplingRate; }
};

} // namespace oversimple
