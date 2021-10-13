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
 * A class to setup an Oversampling object. It contains simple fields to
 * setup how to oversample and if any upsampled audio buffer is needed.
 * Then the Oversampling object will do all the work.
 * @see Oversampling
 */

struct OversamplingSettings
{
  enum class SupportedScalarTypes
  {
    onlyFloat,
    onlyDouble,
    floatAndDouble
  };

  SupportedScalarTypes supportedScalarTypes;
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

  explicit OversamplingSettings(SupportedScalarTypes supportedScalarTypes = SupportedScalarTypes::floatAndDouble,
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
    : supportedScalarTypes(supportedScalarTypes)
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
  int numSamplesPerBlock;
  int rate;

public:
  /**
   * Class to upsample an already interleaved buffer onto an interleaved buffer.
   */
  class VecToVecUpsampler
  {
    std::unique_ptr<IirUpsampler<Scalar>> iirUpsampler;
    std::unique_ptr<TFirUpsampler<Scalar>> firUpsampler;
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
      if (firUpsampler) {
        bool ok = input.deinterleave(firInputBuffer.get(), numChannelsToUpsample, numSamples);
        assert(ok);
        int const numUpsampledSamples =
          firUpsampler->processBlock(firInputBuffer.get(), numChannelsToUpsample, numSamples, firOutputBuffer);
        ok = outputBuffer.interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->processBlock(input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpsampler->getOrder());
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
      int const oversamplingRate = firUpsampler ? (int)firUpsampler->getRate() : (1 << iirUpsampler->getOrder());
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
        firUpsampler = std::make_unique<TFirUpsampler<Scalar>>(settings.numChannels, settings.firTransitionBand);
        firUpsampler->setRate(1 << settings.order);
        iirUpsampler = nullptr;
        firOutputBuffer.setNumChannels(settings.numChannels);
        firInputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<Scalar>::make(settings.numChannels);
        iirUpsampler->setOrder(settings.order);
        firOutputBuffer.setNumChannels(0);
        firInputBuffer.setNumChannels(0);
        firUpsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock);
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
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

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firUpsampler) {
        firUpsampler->reset();
      }
      if (iirUpsampler) {
        iirUpsampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
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

  /**
   * Class to upsample a non interleaved buffer onto an interleaved buffer.
   */
  class ScalarToVecUpsampler
  {
    std::unique_ptr<IirUpsampler<Scalar>> iirUpsampler;
    std::unique_ptr<TFirUpsampler<Scalar>> firUpsampler;
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
      if (firUpsampler) {
        int numUpsampledSamples = firUpsampler->processBlock(input, numChannelsToUpsample, numSamples, firOutputBuffer);
        bool ok = outputBuffer.interleave(firOutputBuffer, 2);
        assert(ok);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->processBlock(input, numSamples, outputBuffer, numChannelsToUpsample);
        return numSamples * (1 << iirUpsampler->getOrder());
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
      int const oversamplingRate = firUpsampler ? (int)firUpsampler->getRate() : (1 << iirUpsampler->getOrder());
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

    /**
     * Create a ScalarToScalarUpsampler object from an OversamplingSettings object.
     */
    ScalarToVecUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<Scalar>>(settings.numChannels, settings.firTransitionBand);
        firUpsampler->setRate(1 << settings.order);
        iirUpsampler = nullptr;
        firOutputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<Scalar>::make(settings.numChannels);
        iirUpsampler->setOrder(settings.order);
        firOutputBuffer.setNumChannels(0);
        firUpsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock);
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firUpsampler) {
        return firUpsampler->getNumSamplesBeforeOutputStarts();
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
      if (firUpsampler) {
        return firUpsampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firUpsampler) {
        firUpsampler->reset();
      }
      if (iirUpsampler) {
        iirUpsampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
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

  /**
   * Class to upsample a non interleaved buffer onto a non interleaved buffer.
   */
  class ScalarToScalarUpsampler
  {
    std::unique_ptr<IirUpsampler<Scalar>> iirUpsampler;
    std::unique_ptr<TFirUpsampler<Scalar>> firUpsampler;
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
      if (firUpsampler) {
        int numUpsampledSamples = firUpsampler->processBlock(input, numChannelsToUpsample, numSamples, outputBuffer);
        return numUpsampledSamples;
      }
      else {
        iirUpsampler->processBlock(input, numSamples, iirOutputBuffer, numChannelsToUpsample);
        iirOutputBuffer.deinterleave(outputBuffer.get(), numChannelsToUpsample, iirOutputBuffer.getNumSamples());
        return numSamples * (1 << iirUpsampler->getOrder());
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
      int const oversamplingRate = firUpsampler ? (int)firUpsampler->getRate() : (1 << iirUpsampler->getOrder());
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

    /**
     * Create a ScalarToScalarUpsampler object from an OversamplingSettings object.
     */
    ScalarToScalarUpsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firUpsampler = std::make_unique<TFirUpsampler<Scalar>>(settings.numChannels, settings.firTransitionBand);
        firUpsampler->setRate(1 << settings.order);
        iirUpsampler = nullptr;
        iirOutputBuffer.setNumChannels(0);
      }
      else {
        iirUpsampler = IirUpsamplerFactory<Scalar>::make(settings.numChannels);
        iirUpsampler->setOrder(settings.order);
        iirOutputBuffer.setNumChannels(settings.numChannels);
        firUpsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock);
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firUpsampler) {
        return firUpsampler->getNumSamplesBeforeOutputStarts();
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
      if (firUpsampler) {
        return firUpsampler->getMaxNumOutputSamples();
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firUpsampler) {
        firUpsampler->reset();
      }
      if (iirUpsampler) {
        iirUpsampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
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

  /**
   * Class to downsample an already interleaved buffer onto a non interleaved buffer.
   */
  struct VecToScalarDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    ScalarBuffer<Scalar> firInputBuffer;

  public:
    /**
     * Create a VecToScalarDownsampler object from an OversamplingSettings object.
     */
    VecToScalarDownsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firDownsampler = std::make_unique<TFirDownsampler<Scalar>>(settings.numChannels, settings.firTransitionBand);
        firDownsampler->setRate(1 << settings.order);
        iirDownsampler = nullptr;
        firInputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirDownsampler = IirDownsamplerFactory<Scalar>::make(settings.numChannels);
        iirDownsampler->setOrder(settings.order);
        firInputBuffer.setNumChannels(0);
        firDownsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock, settings.numSamplesPerBlock * (1 << settings.order));
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownsampler ? (int)firDownsampler->getRate() : (1 << iirDownsampler->getOrder());
      if (firDownsampler) {
        firDownsampler->prepareBuffers(maxNumUpsampledSamples, numSamples);
        firInputBuffer.setNumSamples(maxNumUpsampledSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->prepareBuffer(numSamples);
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
      if (firDownsampler) {
        input.deinterleave(firInputBuffer);
        firDownsampler->processBlock(
          firInputBuffer.get(), numUpsampledSamples, output, numChannelsToDownsample, requiredSamples);
      }
      else {
        iirDownsampler->processBlock(input, requiredSamples * (1 << iirDownsampler->getOrder()));
        iirDownsampler->getOutput().deinterleave(output, numChannelsToDownsample, requiredSamples);
      }
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firDownsampler) {
        return (int)((double)firDownsampler->getNumSamplesBeforeOutputStarts() / (double)firDownsampler->getRate());
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firDownsampler) {
        firDownsampler->reset();
      }
      if (iirDownsampler) {
        iirDownsampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
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

  /**
   * Class to downsample an already interleaved buffer onto an interleaved buffer.
   */
  struct VecToVecDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    ScalarBuffer<Scalar> firInputBuffer;
    ScalarBuffer<Scalar> firOutputBuffer;
    InterleavedBuffer<Scalar> outputBuffer;

  public:
    /**
     * Create a VecToVecDownsampler object from an OversamplingSettings object.
     */
    VecToVecDownsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firDownsampler = std::make_unique<TFirDownsampler<Scalar>>(settings.numChannels, settings.firTransitionBand);
        firDownsampler->setRate(1 << settings.order);
        iirDownsampler = nullptr;
        firInputBuffer.setNumChannels(settings.numChannels);
        firOutputBuffer.setNumChannels(settings.numChannels);
        outputBuffer.setNumChannels(settings.numChannels);
      }
      else {
        iirDownsampler = IirDownsamplerFactory<Scalar>::make(settings.numChannels);
        iirDownsampler->setOrder(settings.order);
        firInputBuffer.setNumChannels(0);
        firOutputBuffer.setNumChannels(0);
        outputBuffer.setNumChannels(0);
        firDownsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock, settings.numSamplesPerBlock * (1 << settings.order));
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownsampler ? (int)firDownsampler->getRate() : (1 << iirDownsampler->getOrder());
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
      if (firDownsampler) {
        input.deinterleave(firInputBuffer.get(), numChannelsToDownsample, numUpsampledSamples);
        firDownsampler->processBlock(
          firInputBuffer.get(), numUpsampledSamples, firOutputBuffer.get(), numChannelsToDownsample, requiredSamples);
        outputBuffer.interleave(firOutputBuffer.get(), numChannelsToDownsample, requiredSamples);
      }
      else {
        iirDownsampler->processBlock(input, requiredSamples * (1 << iirDownsampler->getOrder()));
      }
    }

    InterleavedBuffer<Scalar>& getOutput()
    {
      return firDownsampler ? outputBuffer : iirDownsampler->getOutput();
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firDownsampler) {
        return (int)((double)firDownsampler->getNumSamplesBeforeOutputStarts() / (double)firDownsampler->getRate());
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firDownsampler) {
        firDownsampler->reset();
      }
      if (iirDownsampler) {
        iirDownsampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
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

  /**
   * Class to downsample a non interleaved buffer onto a non interleaved buffer.
   */
  struct ScalarToScalarDownsampler
  {
    std::unique_ptr<IirDownsampler<Scalar>> iirDownsampler;
    std::unique_ptr<TFirDownsampler<Scalar>> firDownsampler;
    InterleavedBuffer<Scalar> iirInputBuffer;

  public:
    /**
     * Create a ScalarToScalarDownsampler object from an OversamplingSettings object.
     */
    ScalarToScalarDownsampler(OversamplingSettings const& settings)
    {
      if (settings.linearPhase) {
        firDownsampler = std::make_unique<TFirDownsampler<Scalar>>(settings.numChannels, settings.firTransitionBand);
        firDownsampler->setRate(1 << settings.order);
        iirDownsampler = nullptr;
        iirInputBuffer.setNumChannels(0);
      }
      else {
        iirDownsampler = IirDownsamplerFactory<Scalar>::make(settings.numChannels);
        iirDownsampler->setOrder(settings.order);
        iirInputBuffer.setNumChannels(2);
        firDownsampler = nullptr;
      }
      prepareBuffers(settings.numSamplesPerBlock, settings.numSamplesPerBlock * (1 << settings.order));
    }

    /**
     * Prepare the resampler to be able to process up to numSamples samples with
     * each processing call.
     * @param numSamples expected number of samples to be processed on each
     * processing call.
     */
    void prepareBuffers(int numSamples, int maxNumUpsampledSamples)
    {
      int const oversamplingRate = firDownsampler ? (int)firDownsampler->getRate() : (1 << iirDownsampler->getOrder());
      if (firDownsampler) {
        firDownsampler->prepareBuffers(maxNumUpsampledSamples, numSamples);
      }
      if (iirDownsampler) {
        iirDownsampler->prepareBuffer(numSamples);
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
      if (firDownsampler) {
        firDownsampler->processBlock(input, numUpsampledSamples, output, numChannelsToDownsample, requiredSamples);
      }
      else {
        iirInputBuffer.interleave(input, numChannelsToDownsample, requiredSamples);
        iirDownsampler->processBlock(iirInputBuffer, requiredSamples * (1 << iirDownsampler->getOrder()));
        iirDownsampler->getOutput().deinterleave(output, numChannelsToDownsample, requiredSamples);
      }
    }

    /**
     * @return the number of input samples needed before a first output sample
     * is produced.
     */
    int getLatency()
    {
      if (firDownsampler) {
        return (int)((double)firDownsampler->getNumSamplesBeforeOutputStarts() / (double)firDownsampler->getRate());
      }
      return 0;
    }

    /**
     * Resets the state of the resampler, clearing its internal buffers.
     */
    void reset()
    {
      if (firDownsampler) {
        firDownsampler->reset();
      }
      if (iirDownsampler) {
        iirDownsampler->reset();
      }
    }

    /**
     * @return the oversampling rate.
     */
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

  /**
   * The requested ScalarToVecUpsampler instances.
   */
  std::vector<std::unique_ptr<ScalarToVecUpsampler>> scalarToVecUpsamplers;

  /**
   * The requested VecToVecUpsampler instances.
   */
  std::vector<std::unique_ptr<VecToVecUpsampler>> vecToVecUpsamplers;

  /**
   * The requested ScalarToScalarUpsampler instances.
   */
  std::vector<std::unique_ptr<ScalarToScalarUpsampler>> scalarToScalarUpsamplers;

  /**
   * The requested VecToScalarDownsampler instances.
   */
  std::vector<std::unique_ptr<VecToScalarDownsampler>> vecToScalarDownsamplers;

  /**
   * The requested VecToVecDownsampler instances.
   */
  std::vector<std::unique_ptr<VecToVecDownsampler>> vecToVecDownsamplers;

  /**
   * The requested ScalarToScalarDownsampler instances.
   */
  std::vector<std::unique_ptr<ScalarToScalarDownsampler>> scalarToScalarDownsamplers;

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
    : numSamplesPerBlock(settings.numSamplesPerBlock)
    , rate(1 << settings.order)
  {
    for (int i = 0; i < settings.numScalarToVecUpsamplers; ++i) {
      scalarToVecUpsamplers.push_back(std::make_unique<ScalarToVecUpsampler>(settings));
    }
    for (int i = 0; i < settings.numVecToVecUpsamplers; ++i) {
      vecToVecUpsamplers.push_back(std::make_unique<VecToVecUpsampler>(settings));
    }
    for (int i = 0; i < settings.numScalarToScalarUpsamplers; ++i) {
      scalarToScalarUpsamplers.push_back(std::make_unique<ScalarToScalarUpsampler>(settings));
    }
    for (int i = 0; i < settings.numVecToScalarDownsamplers; ++i) {
      vecToScalarDownsamplers.push_back(std::make_unique<VecToScalarDownsampler>(settings));
    }
    for (int i = 0; i < settings.numVecToVecDownsamplers; ++i) {
      vecToVecDownsamplers.push_back(std::make_unique<VecToVecDownsampler>(settings));
    }
    for (int i = 0; i < settings.numScalarToScalarDownsamplers; ++i) {
      scalarToScalarDownsamplers.push_back(std::make_unique<ScalarToScalarDownsampler>(settings));
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

  /**
   * Prepare the resampler to be able to process up to numSamples samples with
   * each processing call.
   * @param numSamples expected number of samples to be processed on each
   * processing call.
   */
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
        maxNumUpsampledSamples = scalarToVecUpsamplers[0]->getMaxUpsampledSamples();
      }
      else if (vecToVecUpsamplers.size() > 0) {
        maxNumUpsampledSamples = vecToVecUpsamplers[0]->getMaxUpsampledSamples();
      }
      else if (scalarToScalarUpsamplers.size() > 0) {
        maxNumUpsampledSamples = scalarToScalarUpsamplers[0]->getMaxUpsampledSamples();
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

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
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

  /**
   * Resets the state of the resamplers, clears all the internal buffers.
   */
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
   * OversamplingSettings (numSamplesPerBlock)
   */
  int getNumSamplesPerBlock() const
  {
    return numSamplesPerBlock;
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
    switch (settings.supportedScalarTypes) {
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
      return oversampling32->getLatency();
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
