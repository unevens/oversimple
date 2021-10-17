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
using DownSamplerVecToVec = DownSampler<Scalar>;
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
    : scalarToScalar(maxOrder, numChannels, transitionBand, maxSamplesPerBlock)
  {}

  virtual ~TReSamplerWithConversionBuffer() = default;

  void setMaxOrder(int value)
  {
    scalarToScalar.setMaxOrder(value);
    updateBuffers();
  }

  void setOrder(int value)
  {
    scalarToScalar.setOrder(value);
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value)
  {
    scalarToScalar.setNumChannels(value);
    updateBuffers();
  }

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const
  {
    return scalarToScalar.getNumChannels();
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void prepareBuffers(int numInputSamples)
  {
    scalarToScalar.prepareBuffers(numInputSamples);
    updateBuffers();
  }

  /**
   * Sets the antialiasing filter transition band.
   * @param value the new antialiasing filter transition band, in percentage of
   * the sample rate.
   */
  void setTransitionBand(int value)
  {
    scalarToScalar.setTransitionBand(value);
    updateBuffers();
  }

  /**
   * @return value the antialiasing filter transition band, in percentage of the
   * sample rate.
   */
  double getTransitionBand() const
  {
    return scalarToScalar.getTransitionBand();
  }

  /**
   * Sets the number of samples that will be processed together.
   * @param value the new number of samples that will be processed together.
   */
  void setMaxSamplesPerBlock(int value)
  {
    scalarToScalar.setMaxSamplesPerBlock(value);
    updateBuffers();
  }

  /**
   * @return the number of samples that will be processed together.
   */
  int getMaxSamplesPerBlock() const
  {
    return scalarToScalar.getMaxSamplesPerBlock();
  }

  /**
   * @return the number of input samples needed before a first output sample is
   * produced.
   */
  int getNumSamplesBeforeOutputStarts()
  {
    return scalarToScalar.getNumSamplesBeforeOutputStarts();
  }

  /**
   * @return the maximum number of samples that can be produced by a
   * processBlock call, assuming it is never called with more samples than those
   * passed to prepareBuffers. If prepareBuffers has not been called, then no
   * more samples than maxSamplesPerBlock should be passed to processBlock.
   */
  int getMaxNumOutputSamples() const
  {
    return scalarToScalar.getMaxNumOutputSamples();
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    scalarToScalar.reset();
  }

protected:
  virtual void updateBuffers() = 0;

private:
  Resampler<Scalar> scalarToScalar;
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
    : scalarToScalar(maxOrder, numChannels)
  {}

  virtual ~TReSamplerWithConversionBuffer() = default;

  void setMaxOrder(int value)
  {
    scalarToScalar.setMaxOrder(value);
    updateBuffers();
  }

  void setOrder(int value)
  {
    scalarToScalar.setOrder(value);
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void setNumChannels(int value)
  {
    scalarToScalar.setNumChannels(value);
    updateBuffers();
  }

  /**
   * @return the number of channels the processor is ready to work with.
   */
  int getNumChannels() const
  {
    return scalarToScalar.getNumChannels();
  }

  /**
   * Prepare the processor to work with the supplied number of channels.
   * @param value the new number of channels.
   */
  void prepareBuffers(int numInputSamples)
  {
    maxNumInputSamples = numInputSamples;
    scalarToScalar.prepareBuffers(numInputSamples);
    updateBuffers();
  }

  /**
   * Resets the state of the processor, clearing the buffers.
   */
  void reset()
  {
    scalarToScalar.reset();
  }

protected:
  virtual void updateBuffers() = 0;
  int maxNumInputSamples = 0;
  Resampler<Scalar> scalarToScalar;
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    auto const maxNumOutputSamples = this->scalarToScalar.getMaxNumOutputSamples();
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
    int const numOutputSamples = this->scalarToScalar.processBlock(inputBuffer);
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    auto const maxNumOutputSamples = this->scalarToScalar.getMaxNumOutputSamples();
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
    int const numOutputSamples = this->scalarToScalar.processBlock(inputBuffer);
    return numOutputSamples;
  }

  avec::ScalarBuffer<Scalar>& getOutput()
  {
    return this->scalarToScalar.getOutput();
  }

  avec::ScalarBuffer<Scalar> const& getOutput() const
  {
    return this->scalarToScalar.getOutput();
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->scalarToScalar.getNumChannels();
    auto const maxNumOutputSamples = this->scalarToScalar.getMaxNumOutputSamples();
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
    this->scalarToScalar.processBlock(input, numSamples, scalarOutputBuffer.get(), numOutputChannels, requiredSamples);
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
    this->scalarToScalar.processBlock(input, scalarOutputBuffer.get(), numOutputChannels, requiredSamples);
    auto const ok = outputBuffer.interleave(scalarOutputBuffer, numOutputChannels);
    assert(ok);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->scalarToScalar.getNumChannels();
    auto const maxNumOutputSamples = this->scalarToScalar.getMaxNumOutputSamples();
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
    this->scalarToScalar.processBlock(inputBuffer, outputBuffer.get(), numOutputChannels, requiredSamples);
    auto const ok = output.interleave(outputBuffer, numOutputChannels);
    assert(ok);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->scalarToScalar.getNumChannels();
    auto const maxNumOutputSamples = this->scalarToScalar.getMaxNumOutputSamples();
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
    this->scalarToScalar.processBlock(inputBuffer, output.get(), numOutputChannels, requiredSamples);
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input buffer that holds the input buffer.
   * @return number of upsampled samples
   */
  void processBlock(InterleavedBuffer<Scalar> const& input, Scalar** output, int numOutputChannels, int requiredSamples)
  {
    input.deinterleave(inputBuffer);
    this->scalarToScalar.processBlock(inputBuffer, output, numOutputChannels, requiredSamples);
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->scalarToScalar.getNumChannels();
    auto const maxNumOutputSamples = this->scalarToScalar.getMaxNumOutputSamples();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(maxNumOutputSamples);
  }

  avec::ScalarBuffer<Scalar> inputBuffer;
};
} // namespace fir

namespace iir {

template<typename Scalar>
class UpSamplerScalarToScalar final : public ::oversimple::iir::detail::TUpSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::iir::detail::TUpSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<Scalar> const& input, int numChannels)
  {
    inputBuffer.interleave(input, numChannels);
    int const numOutputSamples = this->scalarToScalar.processBlock(inputBuffer);
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
    inputBuffer.interleave(input, numChannels, numSamples);
    int const numOutputSamples = this->scalarToScalar.processBlock(inputBuffer);
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> inputBuffer;
  avec::ScalarBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class UpSamplerScalarToVec final : public ::oversimple::iir::detail::TUpSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::iir::detail::TUpSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(ScalarBuffer<Scalar> const& input, int numChannels)
  {
    inputBuffer.interleave(input, numChannels);
    int const numOutputSamples = this->scalarToScalar.processBlock(inputBuffer);
    return numOutputSamples;
  }

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(Scalar* const* input, int numChannels, int numSamples)
  {
    inputBuffer.interleave(input, numChannels, numSamples);
    int const numOutputSamples = this->scalarToScalar.processBlock(inputBuffer);
    return numOutputSamples;
  }

  avec::InterleavedBuffer<Scalar>& getOutput()
  {
    return this->get().getOutput();
  }

  avec::InterleavedBuffer<Scalar> const& getOutput() const
  {
    return this->get().getOutput();
  }

private:
  void updateBuffers() override
  {
    auto const numChannels = this->scalarToScalar.getNumChannels();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> inputBuffer;
};

template<typename Scalar>
class UpSamplerVecToScalar final : public ::oversimple::iir::detail::TUpSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::iir::detail::TUpSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

  /**
   * Resamples a multi channel input buffer.
   * @param input ScalarBuffer that holds the input buffer.
   * @return number of upsampled samples
   */
  int processBlock(InterleavedBuffer<Scalar> const& input)
  {
    int const numOutputSamples = this->scalarToScalar.processBlock(input);
    auto const ok = this->get().getOutput().deinterleave(outputBuffer);
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::ScalarBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerScalarToScalar final : public ::oversimple::iir::detail::TDownSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::iir::detail::TDownSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

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
    this->scalarToScalar.processBlock(inputBuffer, numSamples, outputBuffer, numOutputChannels);
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> inputBuffer;
  avec::InterleavedBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerVecToScalar final : public ::oversimple::iir::detail::TDownSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::iir::detail::TDownSamplerWithConversion<Scalar>::TDownSamplerWithConversion;

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
    this->scalarToScalar.processBlock(input, numSamples, outputBuffer, numOutputChannels);
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    outputBuffer.setNumChannels(numChannels);
    outputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> outputBuffer;
};

template<typename Scalar>
class DownSamplerScalarToVec final : public ::oversimple::iir::detail::TDownSamplerWithConversion<Scalar>
{
public:
  using ::oversimple::iir::detail::TDownSamplerWithConversion<Scalar>::TUpSamplerWithConversion;

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
    this->scalarToScalar.processBlock(inputBuffer, numSamples, output, numOutputChannels);
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
    auto const numChannels = this->scalarToScalar.getNumChannels();
    inputBuffer.setNumChannels(numChannels);
    inputBuffer.reserve(this->maxNumInputSamples);
  }

  avec::InterleavedBuffer<Scalar> inputBuffer;
};

} // namespace iir

} // namespace oversimple
