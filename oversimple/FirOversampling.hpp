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
#define NOMINMAX
#include "CDSPResampler.h"
#include "avec/Avec.hpp"

namespace oversimple {

/**
 * Abstract class for FIR resamplers, implementing getters, setters, and filters
 * and buffer management.
 */

class FirResampler
{
protected:
  int oversamplingFactor;
  int numChannels;
  int maxSamplesPerBlock;
  double transitionBand;
  std::vector<std::unique_ptr<r8b::CDSPResampler24>> resamplers;
  int maxOutputLength;
  int maxInputLength;

  virtual void Setup() = 0;
  FirResampler(int numChannels,
               double transitionBand,
               int maxSamplesPerBlock,
               int oversamplingFactor);

public:
  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void SetNumChannels(int value);
  /**
   * @return the number of channels the processor is ready to work with.
   */
  int GetNumChannels() const { return numChannels; }
  /**
   * Sets the overampling factor.
   * @param value the new overampling factor.
   */
  void SetOversamplingFactor(int value);
  /**
   * @return the oversampling factor.
   */
  int GetOversamplingFactor() const { return oversamplingFactor; }
  /**
   * Sets the number of samples that will be processed together.
   * @param value the new number of samples that will be processed together.
   */
  void SetMaxSamplesPerBlock(int value);
  /**
   * @return the number of samples that will be processed together.
   */
  int GetMaxSamplesPerBlock() const { return maxSamplesPerBlock; }
  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void SetTransitionBand(int value);
  /**
   * @return value the antialiasing filter transition band, in percentage of the
   * sample rate.
   */
  double GetTransitionBand() const { return transitionBand; }
  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  int GetNumSamplesBeforeOutputStarts();
  /**
   * @return the maximum number of samples that can be produced by a
   * ProcessBlock call, assuming it is never called with more samples than those
   * passed to PrepareBuffers. If PrepareBuffers has not been called, then no
   * more samples than maxSamplesPerBlock should be passed to ProcessBlock.
   */
  int GetMaxNumOutputSamples() { return maxOutputLength; }
  /**
   * Resets the state of the processor, clearing the buffers.
   */
  virtual void Reset();
  /**
   * Prepare the resampler to be able to process up to numSamples samples with
   * each ProcessBlock call.
   * @param numSamples expected number of samples to be processed on each call
   * to ProcessBlock.
   */
  void PrepareBuffers(int numSamples);
};

/**
 * Upsampler using a FIR antialiasing filter. Only works with double
 * precision input/output.
 * @see TFirUpsampler for a template that can work with single precision.
 */
class FirUpsampler : public FirResampler
{
  void Setup() override;

public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingFactor the oversampling factor
   */
  FirUpsampler(int numChannel,
               double transitionBand = 2.0,
               int maxSamplesPerBlock = 1024,
               int oversamplingFactor = 1);

  /**
   * Upsamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int ProcessBlock(double* const* input,
                   int numChannels,
                   int numSamples,
                   ScalarBuffer<double>& output);

  /**
   * Upsamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int ProcessBlock(ScalarBuffer<double> const& input,
                   ScalarBuffer<double>& output);
};

/**
 * Downsampler using a FIR antialiasing filter. Only works with double
 * precision input/output.
 * @see TFirDownsampler for a template that can work with single precision.
 */
class FirDownsampler : public FirResampler
{
  void Setup() override;
  ScalarBuffer<double> buffer;
  int bufferCounter = 0;
  int maxRequiredOutputLength;

public:
  /**
   * Consructor.
   * @param numChannels the number of channels the processor will be ready to
   * work with.
   * @param transitionBand value the antialiasing filter transition band, in
   * percentage of the sample rate.
   * @param maxSamplesPerBlock the number of samples that will be processed
   * together.
   * @param oversamplingFactor the oversampling factor
   */
  FirDownsampler(int numChannel,
                 double transitionBand = 2.0,
                 int maxSamplesPerBlock = 1024,
                 int oversamplingFactor = 1);

  /**
   * Downsamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void ProcessBlock(ScalarBuffer<double> const& input,
                    double** output,
                    int numChannels,
                    int requiredSamples);

  /**
   * Downsamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void ProcessBlock(ScalarBuffer<double> const& input,
                    ScalarBuffer<double>& output,
                    int requiredSamples);

  void PrepareBuffers(int numInputSamples, int requiredOutputSamples);
  void Reset() override;
};

/**
 * Upsampler using a FIR antialiasing filter.
 */
template<typename Scalar>
class TFirUpsampler final : public FirUpsampler
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
   * @param oversamplingFactor the oversampling factor
   */
  TFirUpsampler(int numChannel,
                double transitionBand = 2.0,
                int maxSamplesPerBlock = 1024,
                int oversamplingFactor = 1)
    : FirUpsampler(numChannel,
                   transitionBand,
                   maxSamplesPerBlock,
                   oversamplingFactor)
  {}
};

template<>
class TFirUpsampler<float> final : public FirUpsampler
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
   * @param oversamplingFactor the oversampling factor
   */
  TFirUpsampler(int numChannel,
                double transitionBand = 2.0,
                int maxSamplesPerBlock = 1024,
                int oversamplingFactor = 1)
    : FirUpsampler(numChannel,
                   transitionBand,
                   maxSamplesPerBlock,
                   oversamplingFactor)
  {}

  /**
   * Upsamples a multi channel input buffer.
   * @param input pointer to the input buffer.
   * @param numChannels number of channels of the input buffer
   * @param numSamples the number of samples of each channel of the input
   * buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int ProcessBlock(float* const* input,
                   int numChannels,
                   int numSamples,
                   ScalarBuffer<float>& output)
  {
    floatToDoubleBuffer.SetNumChannelsAndSize(numChannels, numSamples);
    doubleToFloatBuffer.SetNumChannelsAndSize(numChannels,
                                              numSamples * oversamplingFactor);
    for (int c = 0; c < numChannels; ++c) {
      for (int i = 0; i < numSamples; ++i) {
        floatToDoubleBuffer[c][i] = (double)input[c][i];
      }
    }
    int samples =
      FirUpsampler::ProcessBlock(floatToDoubleBuffer, doubleToFloatBuffer);
    CopyScalarBuffer(doubleToFloatBuffer, output);
    return samples;
  }

  /**
   * Upsamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the upsampled data.
   * @return number of upsampled samples
   */
  int ProcessBlock(ScalarBuffer<float> const& input,
                   ScalarBuffer<float>& output)
  {
    return ProcessBlock(
      input.Get(), input.GetNumChannels(), input.GetSize(), output);
  }
};

/**
 * Downsampler using a FIR antialiasing filter.
 */
template<typename Scalar>
class TFirDownsampler final : public FirDownsampler
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
   * @param oversamplingFactor the oversampling factor
   */
  TFirDownsampler(int numChannel,
                  double transitionBand = 2.0,
                  int maxSamplesPerBlock = 1024,
                  int oversamplingFactor = 1)
    : FirDownsampler(numChannel,
                     transitionBand,
                     maxSamplesPerBlock,
                     oversamplingFactor)
  {}
};

template<>
class TFirDownsampler<float> final : public FirDownsampler
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
   * @param oversamplingFactor the oversampling factor
   */
  TFirDownsampler(int numChannel,
                  double transitionBand = 2.0,
                  int maxSamplesPerBlock = 1024,
                  int oversamplingFactor = 1)
    : FirDownsampler(numChannel,
                     transitionBand,
                     maxSamplesPerBlock,
                     oversamplingFactor)
  {}

  /**
   * Downsamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void ProcessBlock(ScalarBuffer<float> const& input,
                    float** output,
                    int numChannels,
                    int requiredSamples)
  {
    CopyScalarBuffer(input, floatToDoubleBuffer);
    doubleToFloatBuffer.SetNumChannelsAndSize(numChannels, requiredSamples);
    FirDownsampler::ProcessBlock(
      floatToDoubleBuffer, doubleToFloatBuffer, requiredSamples);
    for (int c = 0; c < numChannels; ++c) {
      for (int i = 0; i < requiredSamples; ++i) {
        output[c][i] = (float)doubleToFloatBuffer[c][i];
      }
    }
  }

  /**
   * Downsamples a multi channel input buffer.
   * @param ScalarBuffer that holds the input buffer.
   * @param output ScalarBufffer to hold the downsampled data.
   * @param requiredSamples the number of samples needed as output
   */
  void ProcessBlock(ScalarBuffer<float> const& input,
                    ScalarBuffer<float>& output,
                    int requiredSamples)
  {
    ProcessBlock(input, output.Get(), output.GetNumChannels(), requiredSamples);
  }
};

} // namespace oversimple
