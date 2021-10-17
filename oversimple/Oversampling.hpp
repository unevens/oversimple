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

enum class SupportedScalarTypes
{
  floatAndDouble,
  onlyFloat,
  onlyDouble
};

enum class FilterType
{
  iir,
  fir
};

struct BlockSize final
{
  int numChannels = 2;
  int maxInoutSamples = 256;

#if __cplusplus >= 202002L
  bool operator==(BlockSize const& other) const = default;
#endif
};

/**
 * A class to setup an Oversampling object. It contains simple fields to
 * setup how to oversample and if any upsampled audio buffer is needed.
 * Then the Oversampling object will do all the work.
 * @see Oversampling
 */
struct OversamplingSettings
{
  enum class SupportedScalarTypes
  {
    floatAndDouble,
    onlyFloat,
    onlyDouble,
  };

  struct Requirements
  {
    int order = 0; // the number of stages: 0 for no oversampling, 1 for 2x oversampling, 2 for 4x oversampling...
    int maxOrder = 0;

    int numScalarToVecUpSamplers = 0;
    int numVecToVecUpSamplers = 0;
    int numScalarToScalarUpSamplers = 0;
    int numScalarToScalarDownSamplers = 0;
    int numVecToScalarDownSamplers = 0;
    int numVecToVecDownSamplers = 0;

    int numScalarBuffers = 0;
    int numInterleavedBuffers = 0;

    bool linearPhase = false;
    double firTransitionBand = 4.0;

#if __cplusplus >= 202002L
    bool operator==(Requirements const& other) const = default;
#endif
  };

  struct Context
  {
    SupportedScalarTypes supportedScalarTypes = SupportedScalarTypes::floatAndDouble;
    int numChannels = 2;
    int maxInputSamples = 256;

#if __cplusplus >= 202002L
    bool operator==(Context const& other) const = default;
#endif
  };

  Requirements requirements;
  Context context;

#if __cplusplus >= 202002L
  bool operator==(OversamplingSettings const& other) const = default;
#endif
};

/**
 * A class to abstract over all the implementations in this library, which can
 * be setup with an OversamplingSettings object. It offers a simple api for
 * oversampling and management of upsampled audio buffers. Templated over the sample type.
 * @see OversamplingSettings
 */
template<typename Scalar>
class TOversampling
{
  int maxInputSamples;
  int numChannels;
  int rate;

public:
  /**
   * Class to upsample an already interleaved buffer onto an interleaved buffer.
   */
  class VecToVecUpSampler
  {
    std::unique_ptr<IirUpSampler<Scalar>> iirUpSampler;
    std::unique_ptr<TFirUpSampler<Scalar>> firUpSampler;
    ScalarBuffer<Scalar> firInputBuffer;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    InterleavedBuffer<Scalar>& getOutput()
    {
      return outputBuffer;
    }

    /**
     * Resamples a multi channel input buffer.
     * @param input pointer to the input buffer.
     * @param numChannelsToUpsample number of channels to upsample of the input buffer
     * @param numSamples the number of samples of each channel of the input
     * buffer.
     * @return number of upsampled samples
     */
    int processBlock(InterleavedBuffer<Scalar> const& input, int numChannelsToUpsample, int numSamples)
    {
      if (firUpSampler) {
        bool ok = input.deinterleave(firInputBuffer.get(), numChannelsToUpsample, numSamples);
        assert(ok);
        int const numUpsampledSamples =
          firUpSampler->processBlock(firInputBuffer.get(), numChannelsToUpsample, numSamples, firOutputBuffer);
        ok = outputBuffer.interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpSampler->processBlock(input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpSampler->getOrder());
      }
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples)
    {
      int const oversamplingRate = firUpSampler ? (int)firUpSampler->getRate() : (1 << iirUpSampler->getOrder());
      int const numUpsampledSamples = numSamples * oversamplingRate;

      outputBuffer.setNumSamples(numUpsampledSamples);
      if (firUpSampler) {
        firUpSampler->prepareBuffers(numSamples);
        firInputBuffer.setNumSamples(numSamples);
        firOutputBuffer.setNumSamples(numUpsampledSamples);
      }
      if (iirUpSampler) {
        iirUpSampler->prepareBuffer(numSamples);
      }
    }

    explicit VecToVecUpSampler(OversamplingSettings const& settings)
    {
      if (settings.requirements.linearPhase) {
        firUpSampler = std::make_unique<TFirUpSampler<Scalar>>(settings.context.numChannels,
                                                               settings.requirements.firTransitionBand);
        firUpSampler->setRate(1 << settings.requirements.order);
        iirUpSampler = nullptr;
        firOutputBuffer.setNumChannels(settings.context.numChannels);
        firInputBuffer.setNumChannels(settings.context.numChannels);
      }
      else {
        iirUpSampler = IirUpSamplerFactory<Scalar>::make(settings.context.numChannels);
        iirUpSampler->setOrder(settings.requirements.order);
        firOutputBuffer.setNumChannels(0);
        firInputBuffer.setNumChannels(0);
        firUpSampler = nullptr;
      }
      prepareBuffers(settings.context.maxInputSamples);
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firUpSampler) {
        return firUpSampler->getNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    int getMaxUpsampledSamples()
    {
      if (firUpSampler) {
        return firUpSampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firUpSampler) {
        firUpSampler->reset();
      }
      if (iirUpSampler) {
        iirUpSampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
    int getRate() const
    {
      if (firUpSampler) {
        return firUpSampler->getRate();
      }
      if (iirUpSampler) {
        return 1 << iirUpSampler->getOrder();
      }
    }
  };

  /**
   * Class to upsample a non interleaved buffer onto an interleaved buffer.
   */
  class ScalarToVecUpSampler
  {
    std::unique_ptr<IirUpSampler<Scalar>> iirUpSampler;
    std::unique_ptr<TFirUpSampler<Scalar>> firUpSampler;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    InterleavedBuffer<Scalar>& getOutput()
    {
      return outputBuffer;
    }

    /**
     * Resamples a multi channel input buffer.
     * @param input pointer to the input buffer.
     * @param numChannelsToUpsample number of channels to upsample of the input buffer
     * @param numSamples the number of samples of each channel of the input
     * buffer.
     * @return number of upsampled samples
     */
    int processBlock(Scalar* const* input, int numChannelsToUpsample, int numSamples)
    {
      if (firUpSampler) {
        int numUpsampledSamples = firUpSampler->processBlock(input, numChannelsToUpsample, numSamples, firOutputBuffer);
        bool ok = outputBuffer.interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpSampler->processBlock(input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpSampler->getOrder());
      }
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples)
    {
      int const oversamplingRate = firUpSampler ? (int)firUpSampler->getRate() : (1 << iirUpSampler->getOrder());
      int const numUpsampledSamples = numSamples * oversamplingRate;
      outputBuffer.setNumSamples(numUpsampledSamples);
      if (firUpSampler) {
        firUpSampler->prepareBuffers(numSamples);
        firOutputBuffer.setNumSamples(numUpsampledSamples);
      }
      if (iirUpSampler) {
        iirUpSampler->prepareBuffer(numSamples);
      }
    }

    /**
     * Create a ScalarToScalarUpSampler object from an OversamplingSettings object.
     */
    ScalarToVecUpSampler(OversamplingSettings const& settings)
    {
      if (settings.requirements.linearPhase) {
        firUpSampler = std::make_unique<TFirUpSampler<Scalar>>(settings.context.numChannels,
                                                               settings.requirements.firTransitionBand);
        firUpSampler->setRate(1 << settings.requirements.order);
        iirUpSampler = nullptr;
        firOutputBuffer.setNumChannels(settings.context.numChannels);
      }
      else {
        iirUpSampler = IirUpSamplerFactory<Scalar>::make(settings.context.numChannels);
        iirUpSampler->setOrder(settings.requirements.order);
        firOutputBuffer.setNumChannels(0);
        firUpSampler = nullptr;
      }
      prepareBuffers(settings.context.maxInputSamples);
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firUpSampler) {
        return firUpSampler->getNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    /**
     * @return the maximum number of samples that can be produced by a
     * processBlock call, assuming it is never called with more samples than those
     * passed to prepareBuffers. If prepareBuffers has not been called, then no
     * more samples than maxSamplesPerBlock should be passed to processBlock.
     */
    int getMaxUpsampledSamples()
    {
      if (firUpSampler) {
        return firUpSampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firUpSampler) {
        firUpSampler->reset();
      }
      if (iirUpSampler) {
        iirUpSampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
    int getRate() const
    {
      if (firUpSampler) {
        return firUpSampler->getRate();
      }
      if (iirUpSampler) {
        return 1 << iirUpSampler->getOrder();
      }
    }
  };

  /**
   * Class to upsample a non interleaved buffer onto a non interleaved buffer.
   */
  class ScalarToScalarUpSampler
  {
    std::unique_ptr<IirUpSampler<Scalar>> iirUpSampler;
    std::unique_ptr<TFirUpSampler<Scalar>> firUpSampler;
    ScalarBuffer<Scalar> outputBuffer;
    InterleavedBuffer<Scalar> iirOutputBuffer;

  public:
    ScalarBuffer<Scalar>& getOutput()
    {
      return outputBuffer;
    }

    /**
     * Resamples a multi channel input buffer.
     * @param input pointer to the input buffer.
     * @param numChannelsToUpsample number of channels to upsample of the input buffer
     * @param numSamples the number of samples of each channel of the input
     * buffer.
     * @return number of upsampled samples
     */
    int processBlock(Scalar* const* input, int numChannelsToUpsample, int numSamples)
    {
      if (firUpSampler) {
        int numUpsampledSamples = firUpSampler->processBlock(input, numChannelsToUpsample, numSamples, outputBuffer);
        return numUpsampledSamples;
      }
      else {
        iirUpSampler->processBlock(input, numSamples, iirOutputBuffer, numChannelsToUpsample);
        iirOutputBuffer.deinterleave(outputBuffer.get(), numChannelsToUpsample, iirOutputBuffer.getNumSamples());
        return numSamples * (1 << iirUpSampler->getOrder());
      }
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples)
    {
      int const oversamplingRate = firUpSampler ? (int)firUpSampler->getRate() : (1 << iirUpSampler->getOrder());
      int const numUpsampledSamples = numSamples * oversamplingRate;
      outputBuffer.setNumSamples(numUpsampledSamples);
      if (firUpSampler) {
        firUpSampler->prepareBuffers(numSamples);
      }
      if (iirUpSampler) {
        iirOutputBuffer.setNumSamples(numUpsampledSamples);
        iirUpSampler->prepareBuffer(numSamples);
      }
    }

    /**
     * Create a ScalarToScalarUpSampler object from an OversamplingSettings object.
     */
    ScalarToScalarUpSampler(OversamplingSettings const& settings)
    {
      if (settings.requirements.linearPhase) {
        firUpSampler = std::make_unique<TFirUpSampler<Scalar>>(settings.context.numChannels,
                                                               settings.requirements.firTransitionBand);
        firUpSampler->setRate(1 << settings.requirements.order);
        iirUpSampler = nullptr;
        iirOutputBuffer.setNumChannels(0);
      }
      else {
        iirUpSampler = IirUpSamplerFactory<Scalar>::make(settings.context.numChannels);
        iirUpSampler->setOrder(settings.requirements.order);
        iirOutputBuffer.setNumChannels(settings.context.numChannels);
        firUpSampler = nullptr;
      }
      prepareBuffers(settings.context.maxInputSamples);
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firUpSampler) {
        return firUpSampler->getNumSamplesBeforeOutputStarts();
      }
      return 0;
    }

    /**
     * @return the maximum number of samples that can be produced by a
     * processBlock call, assuming it is never called with more samples than those
     * passed to prepareBuffers. If prepareBuffers has not been called, then no
     * more samples than maxSamplesPerBlock should be passed to processBlock.
     */
    int getMaxUpsampledSamples()
    {
      if (firUpSampler) {
        return firUpSampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firUpSampler) {
        firUpSampler->reset();
      }
      if (iirUpSampler) {
        iirUpSampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
    int getRate() const
    {
      if (firUpSampler) {
        return firUpSampler->getRate();
      }
      if (iirUpSampler) {
        return 1 << iirUpSampler->getOrder();
      }
    }
  };

  /**
   * Class to downsample an already interleaved buffer onto a non interleaved buffer.
   */
  struct VecToScalarDownSampler
  {
    std::unique_ptr<IirDownSampler<Scalar>> iirDownSampler;
    std::unique_ptr<TFirDownSampler<Scalar>> firDownSampler;
    ScalarBuffer<Scalar> firInputBuffer;

  public:
    /**
     * Create a VecToScalarDownSampler object from an OversamplingSettings object.
     */
    VecToScalarDownSampler(OversamplingSettings const& settings)
    {
      if (settings.requirements.linearPhase) {
        firDownSampler = std::make_unique<TFirDownSampler<Scalar>>(settings.context.numChannels,
                                                                   settings.requirements.firTransitionBand);
        firDownSampler->setRate(1 << settings.requirements.order);
        iirDownSampler = nullptr;
        firInputBuffer.setNumChannels(settings.context.numChannels);
      }
      else {
        iirDownSampler = IirDownSamplerFactory<Scalar>::make(settings.context.numChannels);
        iirDownSampler->setOrder(settings.requirements.order);
        firInputBuffer.setNumChannels(0);
        firDownSampler = nullptr;
      }
      prepareBuffers(settings.context.maxInputSamples,
                     settings.context.maxInputSamples * (1 << settings.requirements.order));
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownSampler ? (int)firDownSampler->getRate() : (1 << iirDownSampler->getOrder());
      if (firDownSampler) {
        firDownSampler->prepareBuffers(numSamples);
        firInputBuffer.setNumSamples(maxNumUpsampledSamples);
      }
      if (iirDownSampler) {
        iirDownSampler->prepareBuffer(numSamples);
      }
    }

    /**
     * Resamples a multi channel input buffer.
     * @param input a InterleavedBuffer that holds the input buffer.
     * @param output pointer to the memory in which to store the downsampled data.
     * @param numChannelsToDownsample number of channels to downsample to the output buffer
     * @param numUpsampledSamples the number of input upsampled samples
     * @param requiredSamples the number of samples needed as output
     */
    void processBlock(InterleavedBuffer<Scalar> const& input,
                      Scalar** output,
                      int numChannelsToDownsample,
                      int numUpsampledSamples,
                      int requiredSamples)
    {
      if (firDownSampler) {
        input.deinterleave(firInputBuffer);
        firDownSampler->processBlock(
          firInputBuffer.get(), numUpsampledSamples, output, numChannelsToDownsample, requiredSamples);
      }
      else {
        iirDownSampler->processBlock(input, requiredSamples * (1 << iirDownSampler->getOrder()));
        iirDownSampler->getOutput().deinterleave(output, numChannelsToDownsample, requiredSamples);
      }
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firDownSampler) {
        return (int)((double)firDownSampler->getNumSamplesBeforeOutputStarts() / (double)firDownSampler->getRate());
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firDownSampler) {
        firDownSampler->reset();
      }
      if (iirDownSampler) {
        iirDownSampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
    int getRate() const
    {
      if (firDownSampler) {
        return firDownSampler->getRate();
      }
      if (iirDownSampler) {
        return 1 << iirDownSampler->getOrder();
      }
    }
  };

  /**
   * Class to downsample an already interleaved buffer onto an interleaved buffer.
   */
  struct VecToVecDownSampler
  {
    std::unique_ptr<IirDownSampler<Scalar>> iirDownSampler;
    std::unique_ptr<TFirDownSampler<Scalar>> firDownSampler;
    ScalarBuffer<Scalar> firInputBuffer;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    /**
     * Create a VecToVecDownSampler object from an OversamplingSettings object.
     */
    VecToVecDownSampler(OversamplingSettings const& settings)
    {
      if (settings.requirements.linearPhase) {
        firDownSampler = std::make_unique<TFirDownSampler<Scalar>>(settings.context.numChannels,
                                                                   settings.requirements.firTransitionBand);
        firDownSampler->setRate(1 << settings.requirements.order);
        iirDownSampler = nullptr;
        firInputBuffer.setNumChannels(settings.context.numChannels);
        firOutputBuffer.setNumChannels(settings.context.numChannels);
        outputBuffer.setNumChannels(settings.context.numChannels);
      }
      else {
        iirDownSampler = IirDownSamplerFactory<Scalar>::make(settings.context.numChannels);
        iirDownSampler->setOrder(settings.requirements.order);
        firInputBuffer.setNumChannels(0);
        firOutputBuffer.setNumChannels(0);
        outputBuffer.setNumChannels(0);
        firDownSampler = nullptr;
      }
      prepareBuffers(settings.context.maxInputSamples,
                     settings.context.maxInputSamples * (1 << settings.requirements.order));
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownSampler ? (int)firDownSampler->getRate() : (1 << iirDownSampler->getOrder());
      if (firDownSampler) {
        firDownSampler->prepareBuffers(numSamples);
        firInputBuffer.setNumSamples(maxNumUpsampledSamples);
        firOutputBuffer.setNumSamples(numSamples);
      }
      if (iirDownSampler) {
        iirDownSampler->prepareBuffer(numSamples);
      }
      outputBuffer.setNumSamples(numSamples);
    }

    /**
     * Resamples a multi channel input buffer.
     * @param input a InterleavedBuffer that holds the input buffer.
     * @param numChannelsToDownsample number of channels to downsample to the output buffer
     * @param requiredSamples the number of samples needed as output
     */
    void processBlock(InterleavedBuffer<Scalar> const& input,
                      int numChannelsToDownsample,
                      int numUpsampledSamples,
                      int requiredSamples)
    {
      if (firDownSampler) {
        input.deinterleave(firInputBuffer.get(), numChannelsToDownsample, numUpsampledSamples);
        firDownSampler->processBlock(
          firInputBuffer.get(), numUpsampledSamples, firOutputBuffer.get(), numChannelsToDownsample, requiredSamples);
        outputBuffer.interleave(firOutputBuffer.get(), numChannelsToDownsample, requiredSamples);
      }
      else {
        iirDownSampler->processBlock(input, requiredSamples * (1 << iirDownSampler->getOrder()));
      }
    }

    InterleavedBuffer<Scalar>& getOutput()
    {
      return firDownSampler ? outputBuffer : iirDownSampler->getOutput();
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firDownSampler) {
        return (int)((double)firDownSampler->getNumSamplesBeforeOutputStarts() / (double)firDownSampler->getRate());
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firDownSampler) {
        firDownSampler->reset();
      }
      if (iirDownSampler) {
        iirDownSampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
    int getRate() const
    {
      if (firDownSampler) {
        return firDownSampler->getRate();
      }
      if (iirDownSampler) {
        return 1 << iirDownSampler->getOrder();
      }
    }
  };

  /**
   * Class to downsample a non interleaved buffer onto a non interleaved buffer.
   */
  struct ScalarToScalarDownSampler
  {
    std::unique_ptr<IirDownSampler<Scalar>> iirDownSampler;
    std::unique_ptr<TFirDownSampler<Scalar>> firDownSampler;
    InterleavedBuffer<Scalar> iirInputBuffer;

  public:
    /**
     * Create a ScalarToScalarDownSampler object from an OversamplingSettings object.
     */
    explicit ScalarToScalarDownSampler(OversamplingSettings const& settings)
    {
      if (settings.requirements.linearPhase) {
        firDownSampler = std::make_unique<TFirDownSampler<Scalar>>(settings.context.numChannels,
                                                                   settings.requirements.firTransitionBand);
        firDownSampler->setRate(1 << settings.requirements.order);
        iirDownSampler = nullptr;
        iirInputBuffer.setNumChannels(0);
      }
      else {
        iirDownSampler = IirDownSamplerFactory<Scalar>::make(settings.context.numChannels);
        iirDownSampler->setOrder(settings.requirements.order);
        iirInputBuffer.setNumChannels(2);
        firDownSampler = nullptr;
      }
      prepareBuffers(settings.context.maxInputSamples,
                     settings.context.maxInputSamples * (1 << settings.requirements.order));
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownSampler ? (int)firDownSampler->getRate() : (1 << iirDownSampler->getOrder());
      if (firDownSampler) {
        firDownSampler->prepareBuffers(numSamples);
      }
      if (iirDownSampler) {
        iirDownSampler->prepareBuffer(numSamples);
        iirInputBuffer.setNumSamples(maxNumUpsampledSamples);
      }
    }

    /**
     * Resamples a multi channel input buffer.
     * @param input pointer to the input buffers.
     * @param output pointer to the memory in which to store the downsampled data.
     * @param numChannelsToDownsample number of channels to downsample to the output buffer
     * @param requiredSamples the number of samples needed as output
     */
    void processBlock(Scalar* const* input,
                      Scalar** output,
                      int numChannelsToDownsample,
                      int numUpsampledSamples,
                      int requiredSamples)
    {
      if (firDownSampler) {
        firDownSampler->processBlock(input, numUpsampledSamples, output, numChannelsToDownsample, requiredSamples);
      }
      else {
        iirInputBuffer.interleave(input, numChannelsToDownsample, requiredSamples);
        iirDownSampler->processBlock(iirInputBuffer, requiredSamples * (1 << iirDownSampler->getOrder()));
        iirDownSampler->getOutput().deinterleave(output, numChannelsToDownsample, requiredSamples);
      }
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firDownSampler) {
        return (int)((double)firDownSampler->getNumSamplesBeforeOutputStarts() / (double)firDownSampler->getRate());
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firDownSampler) {
        firDownSampler->reset();
      }
      if (iirDownSampler) {
        iirDownSampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
    int getRate() const
    {
      if (firDownSampler) {
        return firDownSampler->getRate();
      }
      if (iirDownSampler) {
        return 1 << iirDownSampler->getOrder();
      }
    }
  };

  /**
   * The requested ScalarToVecUpSampler instances.
   */
  std::vector<std::unique_ptr<ScalarToVecUpSampler>> scalarToVecUpSamplers;

  /**
   * The requested VecToVecUpSampler instances.
   */
  std::vector<std::unique_ptr<VecToVecUpSampler>> vecToVecUpSamplers;

  /**
   * The requested ScalarToScalarUpSampler instances.
   */
  std::vector<std::unique_ptr<ScalarToScalarUpSampler>> scalarToScalarUpSamplers;

  /**
   * The requested VecToScalarDownSampler instances.
   */
  std::vector<std::unique_ptr<VecToScalarDownSampler>> vecToScalarDownSamplers;

  /**
   * The requested VecToVecDownSampler instances.
   */
  std::vector<std::unique_ptr<VecToVecDownSampler>> vecToVecDownSamplers;

  /**
   * The requested ScalarToScalarDownSampler instances.
   */
  std::vector<std::unique_ptr<ScalarToScalarDownSampler>> scalarToScalarDownSamplers;

  /**
   * The requested InterleavedBuffer instances.
   */
  std::vector<InterleavedBuffer<Scalar>> interleavedBuffers;

  /**
   * The requested ScalarBuffer instances.
   */
  std::vector<ScalarBuffer<Scalar>> scalarBuffers;

  /**
   * Creates a TOversampling object from an OversamplingSettings object.
   * @param settings the settings to initialize from
   */
  TOversampling(OversamplingSettings const& settings)
    : maxInputSamples(settings.context.maxInputSamples)
    , rate(1 << settings.requirements.order)
    , numChannels(settings.context.numChannels)
  {
    for (int i = 0; i < settings.requirements.numScalarToVecUpSamplers; ++i) {
      scalarToVecUpSamplers.push_back(std::make_unique<ScalarToVecUpSampler>(settings));
    }
    for (int i = 0; i < settings.requirements.numVecToVecUpSamplers; ++i) {
      vecToVecUpSamplers.push_back(std::make_unique<VecToVecUpSampler>(settings));
    }
    for (int i = 0; i < settings.requirements.numScalarToScalarUpSamplers; ++i) {
      scalarToScalarUpSamplers.push_back(std::make_unique<ScalarToScalarUpSampler>(settings));
    }
    for (int i = 0; i < settings.requirements.numVecToScalarDownSamplers; ++i) {
      vecToScalarDownSamplers.push_back(std::make_unique<VecToScalarDownSampler>(settings));
    }
    for (int i = 0; i < settings.requirements.numVecToVecDownSamplers; ++i) {
      vecToVecDownSamplers.push_back(std::make_unique<VecToVecDownSampler>(settings));
    }
    for (int i = 0; i < settings.requirements.numScalarToScalarDownSamplers; ++i) {
      scalarToScalarDownSamplers.push_back(std::make_unique<ScalarToScalarDownSampler>(settings));
    }
    int maxNumUpsampledSamples = rate * settings.context.maxInputSamples;
    interleavedBuffers.resize(settings.requirements.numInterleavedBuffers);
    for (auto& buffer : interleavedBuffers) {
      buffer.setNumSamples(maxNumUpsampledSamples);
      buffer.setNumChannels(settings.context.numChannels);
    }
    scalarBuffers.resize(settings.requirements.numScalarBuffers);
    for (auto& buffer : scalarBuffers) {
      buffer.setNumSamples(maxNumUpsampledSamples);
      buffer.setNumChannels(settings.context.numChannels);
    }
  }

  /**
   * Prepare the resampler to be able to process up to numSamples samples with
   * each processing call.
   * @param numSamples expected number of samples to be processed on each
   * processing call.
   */
  void prepareBuffers(int numSamples)
  {
    if (maxInputSamples < numSamples) {

      maxInputSamples = numSamples;

      for (auto& upSampler : scalarToVecUpSamplers) {
        upSampler->prepareBuffers(numSamples);
      }
      for (auto& upSampler : vecToVecUpSamplers) {
        upSampler->prepareBuffers(numSamples);
      }
      for (auto& upSampler : scalarToScalarUpSamplers) {
        upSampler->prepareBuffers(numSamples);
      }

      int maxNumUpsampledSamples = rate * numSamples;

      if (scalarToVecUpSamplers.size() > 0) {
        maxNumUpsampledSamples = scalarToVecUpSamplers[0]->getMaxUpsampledSamples();
      }
      else if (vecToVecUpSamplers.size() > 0) {
        maxNumUpsampledSamples = vecToVecUpSamplers[0]->getMaxUpsampledSamples();
      }
      else if (scalarToScalarUpSamplers.size() > 0) {
        maxNumUpsampledSamples = scalarToScalarUpSamplers[0]->getMaxUpsampledSamples();
      }

      for (auto& downSampler : scalarToScalarDownSamplers) {
        downSampler->prepareBuffers(numSamples, maxNumUpsampledSamples);
      }
      for (auto& downSampler : vecToScalarDownSamplers) {
        downSampler->prepareBuffers(numSamples, maxNumUpsampledSamples);
      }
      for (auto& downSampler : vecToVecDownSamplers) {
        downSampler->prepareBuffers(numSamples, maxNumUpsampledSamples);
      }

      for (auto& buffer : interleavedBuffers) {
        buffer.setNumSamples(maxNumUpsampledSamples);
      }

      for (auto& buffer : scalarBuffers) {
        buffer.setNumSamples(maxNumUpsampledSamples);
      }
    }
  }

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  int getLatency()
  {
    int upsamplingLatency = 0;
    if (scalarToScalarUpSamplers.size() > 0) {
      upsamplingLatency = scalarToScalarUpSamplers[0]->getLatency();
    }
    if (scalarToVecUpSamplers.size() > 0) {
      upsamplingLatency = std::max(upsamplingLatency, scalarToVecUpSamplers[0]->getLatency());
    }
    if (vecToVecUpSamplers.size() > 0) {
      upsamplingLatency = std::max(upsamplingLatency, vecToVecUpSamplers[0]->getLatency());
    }
    int downsamplingLatency = 0;
    if (vecToScalarDownSamplers.size() > 0) {
      downsamplingLatency = vecToScalarDownSamplers[0]->getLatency();
    }
    if (scalarToScalarDownSamplers.size() > 0) {
      downsamplingLatency = std::max(downsamplingLatency, scalarToScalarDownSamplers[0]->getLatency());
    }
    if (vecToVecDownSamplers.size() > 0) {
      downsamplingLatency = std::max(downsamplingLatency, vecToVecDownSamplers[0]->getLatency());
    }
    auto const latency = upsamplingLatency + downsamplingLatency;
    return latency;
  }

  /**
   * Resets the state of the resamplers, clears all the internal buffers.
   */
  void reset()
  {
    for (auto& upSampler : scalarToVecUpSamplers) {
      upSampler->reset();
    }
    for (auto& upSampler : vecToVecUpSamplers) {
      upSampler->reset();
    }
    for (auto& downSampler : scalarToScalarDownSamplers) {
      downSampler->reset();
    }
    for (auto& downSampler : vecToVecDownSamplers) {
      downSampler->reset();
    }
    for (auto& downSampler : vecToScalarDownSamplers) {
      downSampler->reset();
    }
  }

  /**
   * @return the oversampling rate.
   */
  int getRate() const
  {
    return rate;
  }

  /**
   * @return the maximum number of samples that can be processed by a single
   * process call. It is the same number passed to prepareBuffers or set in the
   * OversamplingSettings (maxInputSamples)
   */
  int getmaxInputSamples() const
  {
    return maxInputSamples;
  }

  /**
   * @return the maximum number of channels supported
   */
  int getNumChannels() const
  {
    return numChannels;
  }
};

/**
 * A class that holds TOversampling instances for float and double,
 */
class Oversampling final
{
public:
  /**
   * Creates a Oversampling object from an OversamplingSettings object.
   * @param settings the settings to initialize from
   */
  explicit Oversampling(OversamplingSettings const& settings)
  {
    switch (settings.context.supportedScalarTypes) {
      case OversamplingSettings::SupportedScalarTypes::floatAndDouble:
        oversampling32 = std::make_unique<TOversampling<float>>(settings);
        oversampling64 = std::make_unique<TOversampling<double>>(settings);
        break;
      case OversamplingSettings::SupportedScalarTypes::onlyFloat:
        oversampling32 = std::make_unique<TOversampling<float>>(settings);
        break;
      case OversamplingSettings::SupportedScalarTypes::onlyDouble:
        oversampling64 = std::make_unique<TOversampling<double>>(settings);
        break;
    }
  }

  /**
   * @return the oversampling object relative to the Scalar type
   */
  template<class Scalar>
  TOversampling<Scalar>& get();

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  int getLatency() const
  {
    if (oversampling64)
      return oversampling64->getLatency();
    else
      return oversampling32->getLatency();
  }

  /**
   * @return the oversampling rate.
   */
  int getRate() const
  {
    if (oversampling64)
      return oversampling64->getRate();
    else
      return oversampling32->getRate();
  }

  /**
   * @return the maximum number of channels supported
   */
  int getNumChannels() const
  {
    if (oversampling64)
      return oversampling64->getNumChannels();
    else
      return oversampling32->getNumChannels();
  }

private:
  std::unique_ptr<TOversampling<float>> oversampling32;
  std::unique_ptr<TOversampling<double>> oversampling64;
};

template<>
inline TOversampling<float>& Oversampling::get<float>()
{
  return *oversampling32;
}

template<>
inline TOversampling<double>& Oversampling::get<double>()
{
  return *oversampling64;
}

} // namespace oversimple
