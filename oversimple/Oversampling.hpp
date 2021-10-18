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
#include "oversimple/FirOversampling.hpp"
#include "oversimple/IirOversamplingFactory.hpp"
#include <functional>

namespace oversimple {

namespace fir {
template<typename Scalar>
using UpSamplerScalarToScalar = detail::TUpSamplerPreAllocated<Scalar>;

template<typename Scalar>
using DownSamplerScalarToScalar = detail::TDownSamplerPreAllocated<Scalar>;
} // namespace fir

namespace iir {

template<typename Scalar>
using UpSamplerVecToVec = UpSampler<Scalar>;

template<typename Scalar>
using UpSamplerAnyToVec = UpSampler<Scalar>;
} // namespace iir

namespace detail {} // namespace detail

namespace fir::detail {
template<typename Scalar, template<class> class Resampler>
class TReSamplerWithConversionBuffer
{
public:
  explicit TReSamplerWithConversionBuffer(int maxOrder = 5,
                                          int numChannels = 2,
                                          double transitionBand = 2.0,
                                          int maxSamplesPerBlock = 1024)
    : resampler(maxOrder, numChannels, transitionBand, maxSamplesPerBlock)
  {}

  virtual ~TReSamplerWithConversionBuffer() = default;

  void setMaxOrder(int value)
  {
    resampler.setMaxOrder(value);
    updateBuffers();
  }

  void setOrder(int value)
  {
    resampler.setOrder(value);
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value)
  {
    resampler.setNumChannels(value);
    updateBuffers();
  }

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const
  {
    return resampler.getNumChannels();
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void prepareBuffers(int numInputSamples)
  {
    resampler.prepareBuffers(numInputSamples);
    updateBuffers();
  }

  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setTransitionBand(int value)
  {
    resampler.setTransitionBand(value);
    updateBuffers();
  }

  /**
   * @return value the antialiasing filter transition band, in percentage of the
   * sample rate.
   */
  double getTransitionBand() const
  {
    return resampler.getTransitionBand();
  }

  /**
   * Sets the number of samples that will be processed together.
   * @param value the new number of samples that will be processed together.
   */
  void setMaxSamplesPerBlock(int value)
  {
    resampler.setMaxSamplesPerBlock(value);
    updateBuffers();
  }

  /**
   * @return the number of samples that will be processed together.
   */
  int getMaxSamplesPerBlock() const
  {
    return resampler.getMaxSamplesPerBlock();
  }

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  int getNumSamplesBeforeOutputStarts()
  {
    return resampler.getNumSamplesBeforeOutputStarts();
  }

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers. If prepareBuffers has not been called, then no
   * more samples than maxSamplesPerBlock should be passed to processBlock.
   */
  int getMaxNumOutputSamples() const
  {
    return resampler.getMaxNumOutputSamples();
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    resampler.reset();
  }

protected:
  virtual void updateBuffers() = 0;

private:
  Resampler<Scalar> resampler;
};

template<typename Scalar>
using TUpSamplerWithConversion = TReSamplerWithConversionBuffer<Scalar, fir::detail::TUpSamplerPreAllocated>;

template<typename Scalar>
using TDownSamplerWithConversion = TReSamplerWithConversionBuffer<Scalar, fir::detail::TDownSamplerPreAllocated>;

} // namespace fir::detail

namespace iir::detail {
template<typename Scalar, template<class> class Resampler>
class TReSamplerWithConversionBuffer
{
public:
  explicit TReSamplerWithConversionBuffer(int maxOrder = 5, int numChannels = 2)
    : resampler(maxOrder, numChannels)
  {}

  virtual ~TReSamplerWithConversionBuffer() = default;

  void setMaxOrder(int value)
  {
    resampler.setMaxOrder(value);
    updateBuffers();
  }

  void setOrder(int value)
  {
    resampler.setOrder(value);
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value)
  {
    resampler.setNumChannels(value);
    updateBuffers();
  }

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const
  {
    return resampler.getNumChannels();
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void prepareBuffers(int numInputSamples)
  {
    maxNumInputSamples = numInputSamples;
    resampler.prepareBuffers(numInputSamples);
    updateBuffers();
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    resampler.reset();
  }

protected:
  virtual void updateBuffers() = 0;
  int maxNumInputSamples = 0;
  Resampler<Scalar> resampler;
};

template<typename Scalar>
using TUpSamplerWithConversion = TReSamplerWithConversionBuffer<Scalar, UpSampler>;

template<typename Scalar>
using TDownSamplerWithConversion = TReSamplerWithConversionBuffer<Scalar, DownSampler>;
} // namespace iir::detail

namespace fir {
template<typename Scalar>
class UpSamplerScalarToVec final : public ::oversimple::fir::detail::TUpSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::fir::detail::TUpSamplerWithConversion<Scalar>::TUpSampleWithConversion;

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
    int const numOutputSamples = this->get().processBlock(input, numChannels, numSamples);
    auto const ok = outputBuffer.interleave(this->get().getOutput(), numChannels);
    assert(ok);
    return numOutputSamples;
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<double> const& input)
  {
    int const numOutputSamples = this->get().processBlock(input);
    auto const ok = outputBuffer.interleave(this->get().getOutput(), input.getNumChannels());
    assert(ok);
    return numOutputSamples;
  }

  avec::InterleavedBuffer<Scalar>& getOutput()
  {
    return outputBuffer;
  }

  avec::InterleavedBuffer<Scalar> const& getOutput() const
  {
    return outputBuffer;
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    auto const maxNumOutputSamples = this->resampler.getMaxNumOutputSamples();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(maxNumOutputSamples);
  }

  avec::InterleavedBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class UpSamplerVecToVec final : public ::oversimple::fir::detail::TUpSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::fir::detail::TUpSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(InterleavedBuffer<Scalar> const& input)
  {
    input.deinterleave(inputBuffer);
    int const numOutputSamples = this->resampler.processBlock(inputBuffer);
    auto const ok = outputBuffer.interleave(this->get().getOutput(), inputBuffer.getNumChannels());
    assert(ok);
    return numOutputSamples;
  }

  avec::InterleavedBuffer<Scalar>& getOutput()
  {
    return outputBuffer;
  }

  avec::InterleavedBuffer<Scalar> const& getOutput() const
  {
    return outputBuffer;
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    auto const maxNumOutputSamples = this->resampler.getMaxNumOutputSamples();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(maxNumOutputSamples);
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(maxNumOutputSamples);
  }

  avec::InterleavedBuffer<Scalar> outputBuffer;
  avec::ScalarBuffer<Scalar> inputBuffer;
};

template<typename Scalar>
class UpSamplerVecToScalar final : public ::oversimple::fir::detail::TUpSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::fir::detail::TUpSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(InterleavedBuffer<Scalar> const& input)
  {
    input.deinterleave(inputBuffer);
    int const numOutputSamples = this->resampler.processBlock(inputBuffer);
    return numOutputSamples;
  }

  avec::ScalarBuffer<Scalar>& getOutput()
  {
    return this->resampler.getOutput();
  }

  avec::ScalarBuffer<Scalar> const& getOutput() const
  {
    return this->resampler.getOutput();
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    auto const maxNumOutputSamples = this->resampler.getMaxNumOutputSamples();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(maxNumOutputSamples);
  }

  avec::ScalarBuffer<Scalar> inputBuffer;
};

template<typename Scalar>
class DownSamplerScalarToVec final : public ::oversimple::fir::detail::TDownSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::fir::detail::TDownSamplerWithConversion<Scalar>::TDownSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output interleaved buffer for the output
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Scalar* const* input,
                    int numSamples,
                    InterleavedBuffer<Scalar>& output,
                    int numOutputChannels,
                    int requiredSamples)
  {
    this->resampler.processBlock(input, numSamples, scalarOutputBuffer.get(), numOutputChannels, requiredSamples);
    auto const ok = outputBuffer.interleave(scalarOutputBuffer, numOutputChannels);
    assert(ok);
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input a ScalarBuffer that holds the input buffer.
   * @param output pointer to the memory in which to store the downsampled data.
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(ScalarBuffer<Scalar> const& input,
                    InterleavedBuffer<Scalar>& output,
                    int numOutputChannels,
                    int requiredSamples)
  {
    this->resampler.processBlock(input, scalarOutputBuffer.get(), numOutputChannels, requiredSamples);
    auto const ok = outputBuffer.interleave(scalarOutputBuffer, numOutputChannels);
    assert(ok);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    auto const maxNumOutputSamples = this->resampler.getMaxNumOutputSamples();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(maxNumOutputSamples);
    scalarOutputBuffer.setNumChannels(numChannels);
    scalarOutputBuffer.reserve(maxNumOutputSamples);
  }

  avec::InterleavedBuffer<Scalar> outputBuffer;
  avec::ScalarBuffer<Scalar> scalarOutputBuffer;
};

template<typename Scalar>
class DownSamplerVecToVec final : public ::oversimple::fir::detail::TDownSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::fir::detail::TDownSamplerWithConversion<Scalar>::TDownSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  void processBlock(InterleavedBuffer<Scalar> const& input,
                    InterleavedBuffer<Scalar>& output,
                    int numOutputChannels,
                    int requiredSamples)
  {
    input.deinterleave(inputBuffer);
    this->resampler.processBlock(inputBuffer, outputBuffer.get(), numOutputChannels, requiredSamples);
    auto const ok = output.interleave(outputBuffer, numOutputChannels);
    assert(ok);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    auto const maxNumOutputSamples = this->resampler.getMaxNumOutputSamples();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(maxNumOutputSamples);
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(maxNumOutputSamples);
  }

  avec::ScalarBuffer<Scalar> inputBuffer;
  avec::ScalarBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerVecToScalar final : public ::oversimple::fir::detail::TDownSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::fir::detail::TDownSamplerWithConversion<Scalar>::TDownSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  void processBlock(InterleavedBuffer<Scalar> const& input,
                    ScalarBuffer<Scalar>& output,
                    int numOutputChannels,
                    int requiredSamples)
  {
    input.deinterleave(inputBuffer);
    this->resampler.processBlock(inputBuffer, output.get(), numOutputChannels, requiredSamples);
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input buffer that holds the input buffer.
   * @return number of upsampled samples
   */
  void processBlock(InterleavedBuffer<Scalar> const& input, Scalar** output, int numOutputChannels, int requiredSamples)
  {
    input.deinterleave(inputBuffer);
    this->resampler.processBlock(inputBuffer, output, numOutputChannels, requiredSamples);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    auto const maxNumOutputSamples = this->resampler.getMaxNumOutputSamples();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(maxNumOutputSamples);
  }

  avec::ScalarBuffer<Scalar> inputBuffer;
};
} // namespace fir

namespace iir {

template<typename Scalar>
class UpSamplerAnyToScalar final : public detail::TUpSamplerWithConversion<Scalar>
{
public:
  using detail::TUpSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(InterleavedBuffer<Scalar> const& input)
  {
    int const numOutputSamples = this->resampler.processBlock(input);
    auto const ok = this->get().getOutput().deinterleave(outputBuffer);
    assert(ok);
    return numOutputSamples;
  }

  int processBlock(ScalarBuffer<Scalar> const& input, int numChannels)
  {
    int const numOutputSamples = this->resampler.processBlock(input, numChannels);
    auto const ok = this->get().getOutput().deinterleave(outputBuffer, numChannels);
    assert(ok);
    return numOutputSamples;
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(Scalar* const* input, int numChannels, int numSamples)
  {
    int const numOutputSamples = this->resampler.processBlock(input, numChannels, numSamples);
    auto const ok = this->get().getOutput().deinterleave(outputBuffer, numChannels);
    assert(ok);
    return numOutputSamples;
  }

  avec::InterleavedBuffer<Scalar>& getOutput()
  {
    return outputBuffer;
  }

  avec::InterleavedBuffer<Scalar> const& getOutput() const
  {
    return outputBuffer;
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::ScalarBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerScalarToScalar final : public detail::TDownSamplerWithConversion<Scalar>
{
public:
  using detail::TDownSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output interleaved buffer for the output
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Scalar* const* input, int numSamples, Scalar** output, int numOutputChannels, int unused = 0)
  {
    inputBuffer.interleave(input, numOutputChannels);
    this->resampler.processBlock(inputBuffer, numSamples, outputBuffer, numOutputChannels);
    outputBuffer.deinterleave(output, numOutputChannels);
  }

  void processBlock(avec::ScalarBuffer<Scalar>& input,
                    int numSamples,
                    avec::ScalarBuffer<Scalar>& output,
                    int numOutputChannels,
                    int requiredSamples,
                    int unused = 0)
  {
    processBlock(input.get(), numSamples, output.get(), numOutputChannels, requiredSamples);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> inputBuffer;
  avec::InterleavedBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerVecToScalar final : public detail::TDownSamplerWithConversion<Scalar>
{
public:
  using detail::TDownSamplerWithConversion<Scalar>::TDownSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output interleaved buffer for the output
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(InterleavedBuffer<Scalar> const& input,
                    int numSamples,
                    Scalar** output,
                    int numOutputChannels,
                    int unused = 0)
  {
    this->resampler.processBlock(input, numSamples, outputBuffer, numOutputChannels);
    outputBuffer.deinterleave(output, numOutputChannels);
  }

  void processBlock(avec::ScalarBuffer<Scalar>& input,
                    int numSamples,
                    avec::ScalarBuffer<Scalar>& output,
                    int numOutputChannels,
                    int unused = 0)
  {
    processBlock(input.get(), numSamples, output.get(), numOutputChannels);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerScalarToVec final : public detail::TDownSamplerWithConversion<Scalar>
{
public:
  using detail::TDownSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input pointer to the input buffers.
   * @param output interleaved buffer for the output
   * @param numOutputChannels number of channels of the output buffer
   * @param requiredSamples the number of samples needed as output
   */
  void processBlock(Scalar* const* input,
                    int numSamples,
                    avec::InterleavedBuffer<Scalar>& output,
                    int numOutputChannels,
                    int unused = 0)
  {
    inputBuffer.interleave(input, numOutputChannels);
    this->resampler.processBlock(inputBuffer, numSamples, output, numOutputChannels);
  }

  void processBlock(avec::ScalarBuffer<Scalar>& input,
                    int numSamples,
                    avec::InterleavedBuffer<Scalar>& output,
                    int numOutputChannels,
                    int requiredSamples,
                    int unused = 0)
  {
    processBlock(input.get(), numSamples, output, numOutputChannels, requiredSamples);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->resampler.getNumChannels();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> inputBuffer;
};

} // namespace iir

} // namespace oversimple
