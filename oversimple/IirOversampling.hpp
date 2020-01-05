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

#include "avec/Avec.hpp"
#include "hiir/Downsampler2x2SseDouble.h"
#include "hiir/Downsampler2x4Sse.h"
#include "hiir/Upsampler2x2SseDouble.h"
#include "hiir/Upsampler2x4Sse.h"
#include "oversimple/IirOversamplingDesigner.hpp"

#if __AVX__
#include "hiir/Downsampler2x4AvxDouble.h"
#include "hiir/Downsampler2x8Avx.h"
#include "hiir/Upsampler2x4AvxDouble.h"
#include "hiir/Upsampler2x8Avx.h"
#else
namespace hiir {
struct FakeInterface
{
  void clear_buffers() {}
  void set_coefs(const double coef_arr[]) {}
  void process_block(double out_ptr[], const double in_ptr[], long nbr_spl) {}
};
template<int NC>
class Downsampler2x4AvxDouble final : public FakeInterface
{};
template<int NC>
class Downsampler2x8Avx final : public FakeInterface
{};
template<int NC>
class Upsampler2x4AvxDouble final : public FakeInterface
{};
template<int NC>
class Upsampler2x8Avx final : public FakeInterface
{};
} // namespace hiir
#endif

namespace oversimple {

// interfaces

/**
 * Abstract class for IIR resamplers.
 */
class IirOversampler
{
public:
  /**
   * @return the oversampling order
   */
  virtual int GetOrder() const = 0;
  /**
   * Sets the oversampling order
   * @param value the new oversampling order.
   */
  virtual void SetOrder(int value) = 0;
  /**
   * Preallocates data to process the supplied amount of samples.
   * @param samplesPerBlock the number of samples to preallocate memory for
   */
  virtual void PrepareBuffer(int samplesPerBlock) = 0;
  /**
   * Sets the number of channels the processor is capable to work with.
   * @param value the new number of channels.
   */
  virtual void SetNumChannels(int value) = 0;
  /**
   * Resets the state of the processor, clearing the state of the antialiasing
   * filters.
   */
  virtual void Reset() = 0;
  /**
   * @return the IirOversamplingDesigner used to create the processor.
   */
  virtual IirOversamplingDesigner const& GetDesigner() const = 0;
  virtual ~IirOversampler() {}
};

/**
 * Abstract class for IIR upsamplers.
 */
template<typename Scalar>
class IirUpsampler : public virtual IirOversampler
{
public:
  /**
   * Upsamples the input.
   * @param input a ScalarBuffer that holds the input samples
   * @param output an InterleavedBuffer to hold the upsampled samples
   */
  virtual void ProcessBlock(ScalarBuffer<Scalar> const& input,
                            InterleavedBuffer<Scalar>& output) = 0;

  /**
   * Upsamples the input.
   * @param input a pointer to the memory holding the input samples
   * @param numInputSamples the number of samples in each channel of the input
   * buffer
   * @param output an InterleavedBuffer to hold the upsampled samples
   */
  virtual void ProcessBlock(Scalar* const* input,
                            int numInputSamples,
                            InterleavedBuffer<Scalar>& output) = 0;
};

/**
 * Abstract class for IIR downsamplers.
 */
template<typename Scalar>
class IirDownsampler : public virtual IirOversampler
{
public:
  /**
   * Downsamples the input.
   * @param input InterleavedBuffer holding the interleaved input samples.
   * @param numSamples the number of samples to donwsample from each channel of
   * the input
   */
  virtual void ProcessBlock(InterleavedBuffer<Scalar>& input,
                            int numSamples) = 0;

  /**
   * @return a reference to the InterleavedBuffer holding the donwsampled
   * samples.
   */
  virtual InterleavedBuffer<Scalar>& GetOutput() = 0;
};

// implementations

namespace {

template<typename Scalar,
         int numCoefsStage0,
         int numCoefsStage1,
         int numCoefsStage2,
         int numCoefsStage3,
         template<int>
         class StageVec8,
         template<int>
         class StageVec4,
         template<int>
         class StageVec2>
class IirOversamplingChain : public virtual IirOversampler
{
protected:
  static constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;

  static constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;

  template<class T>
  using aligned_vector = aligned_vector<T>;

  aligned_vector<StageVec8<numCoefsStage0>> stage8_0;
  aligned_vector<StageVec8<numCoefsStage1>> stage8_1;
  aligned_vector<StageVec8<numCoefsStage2>> stage8_2;
  aligned_vector<StageVec8<numCoefsStage3>> stage8_3;

  aligned_vector<StageVec4<numCoefsStage0>> stage4_0;
  aligned_vector<StageVec4<numCoefsStage1>> stage4_1;
  aligned_vector<StageVec4<numCoefsStage2>> stage4_2;
  aligned_vector<StageVec4<numCoefsStage3>> stage4_3;

  aligned_vector<StageVec2<numCoefsStage0>> stage2_0;
  aligned_vector<StageVec2<numCoefsStage1>> stage2_1;
  aligned_vector<StageVec2<numCoefsStage2>> stage2_2;
  aligned_vector<StageVec2<numCoefsStage3>> stage2_3;

  IirOversamplingDesigner designer;
  int numChannels;
  int order;
  int factor;
  int samplesPerBlock;
  InterleavedBuffer<Scalar> buffer[2];

  IirOversamplingChain(IirOversamplingDesigner designer_, int numChannels_)
    : designer(designer_)
    , numChannels(numChannels_)
    , samplesPerBlock(256)
    , order(0)
    , factor(1)
  {
    assert(designer.GetStages().size() == 4);
    SetupBuffer();
    SetupStages();
  }

  void SetupStages()
  {
    if constexpr (VEC8_AVAILABLE) {
      auto d = std::div(numChannels, 8);
#if AVEC_MIX_VEC_SIZES
      int numOf8 = d.quot + (d.rem > 4 ? 1 : 0);
      stage8_0.resize(numOf8);
      stage8_1.resize(numOf8);
      stage8_2.resize(numOf8);
      stage8_3.resize(numOf8);
      if (d.rem > 0 && d.rem <= 4) {
        stage4_0.resize(1);
        stage4_1.resize(1);
        stage4_2.resize(1);
        stage4_3.resize(1);
      }
#else
      int numOf8 = d.quot + (d.rem > 0 ? 1 : 0);
      stage8_0.resize(numOf8);
      stage8_1.resize(numOf8);
      stage8_2.resize(numOf8);
      stage8_3.resize(numOf8);
#endif
    }
    else if constexpr (VEC4_AVAILABLE) {
      auto d = std::div(numChannels, 4);
      int numProcessorsForStage = d.quot + (d.rem > 0 ? 1 : 0);
      stage4_0.resize(numProcessorsForStage);
      stage4_1.resize(numProcessorsForStage);
      stage4_2.resize(numProcessorsForStage);
      stage4_3.resize(numProcessorsForStage);
      stage8_0.resize(0);
      stage8_1.resize(0);
      stage8_2.resize(0);
      stage8_3.resize(0);
    }
    else {
      auto d = std::div(numChannels, 2);
      int numProcessorsForStage = d.quot + (d.rem > 0 ? 1 : 0);
      stage2_0.resize(numProcessorsForStage);
      stage2_1.resize(numProcessorsForStage);
      stage2_2.resize(numProcessorsForStage);
      stage2_3.resize(numProcessorsForStage);
    }

    auto& stages = designer.GetStages();
    std::vector<double> coefs;

    stages[0].ComputeCoefs(coefs);
    for (auto& stage : stage4_0) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage8_0) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage2_0) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    stages[1].ComputeCoefs(coefs);
    for (auto& stage : stage4_1) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage8_1) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage2_1) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    stages[2].ComputeCoefs(coefs);
    for (auto& stage : stage4_2) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage8_2) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage2_2) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    stages[3].ComputeCoefs(coefs);
    for (auto& stage : stage4_3) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage8_3) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage2_3) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
  }

  void SetupBuffer()
  {
    for (int i = 0; i < 2; ++i) {
      buffer[i].SetNumChannels(numChannels);
      buffer[i].SetNumSamples(samplesPerBlock * factor);
    }
  }

  IirOversamplingDesigner const& GetDesigner() const override
  {
    return designer;
  }
  int GetOrder() const override { return order; }
  void SetOrder(int value) override
  {
    assert(order > -1 && order < 5);
    order = value;
    factor = 1 << order;
    SetupBuffer();
  }
  void PrepareBuffer(int samplesPerBlock_) override
  {
    samplesPerBlock = samplesPerBlock_;
    SetupBuffer();
  }
  void SetNumChannels(int value) override
  {
    numChannels = value;
    SetupBuffer();
    SetupStages();
  }

  void ApplyStage0(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar>& input,
                   int numSamples)
  {
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_0) {
        auto& out = output.GetBuffer8(i);
        auto& in = input.GetBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_0) {
        auto& out = output.GetBuffer4(i);
        auto& in = input.GetBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    else {
      int i = 0;
      for (auto& stage : stage2_0) {
        auto& out = output.GetBuffer2(i);
        auto& in = input.GetBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
  }

  void ApplyStage1(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar>& input,
                   int numSamples)
  {
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_1) {
        auto& out = output.GetBuffer8(i);
        auto& in = input.GetBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_1) {
        auto& out = output.GetBuffer4(i);
        auto& in = input.GetBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    else {
      int i = 0;
      for (auto& stage : stage2_1) {
        auto& out = output.GetBuffer2(i);
        auto& in = input.GetBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
  }

  void ApplyStage2(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar>& input,
                   int numSamples)
  {

    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_2) {
        auto& out = output.GetBuffer8(i);
        auto& in = input.GetBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_2) {
        auto& out = output.GetBuffer4(i);
        auto& in = input.GetBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    else {
      int i = 0;
      for (auto& stage : stage2_2) {
        auto& out = output.GetBuffer2(i);
        auto& in = input.GetBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
  }

  void ApplyStage3(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar>& input,
                   int numSamples)
  {
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_3) {
        auto& out = output.GetBuffer8(i);
        auto& in = input.GetBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_3) {
        auto& out = output.GetBuffer4(i);
        auto& in = input.GetBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
    else {
      int i = 0;
      for (auto& stage : stage2_3) {
        auto& out = output.GetBuffer2(i);
        auto& in = input.GetBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
      }
    }
  }

  void Reset() override
  {
    for (auto& stage : stage8_0) {
      stage.clear_buffers();
    }
    for (auto& stage : stage8_1) {
      stage.clear_buffers();
    }
    for (auto& stage : stage8_2) {
      stage.clear_buffers();
    }
    for (auto& stage : stage8_3) {
      stage.clear_buffers();
    }
    for (auto& stage : stage4_0) {
      stage.clear_buffers();
    }
    for (auto& stage : stage4_1) {
      stage.clear_buffers();
    }
    for (auto& stage : stage4_2) {
      stage.clear_buffers();
    }
    for (auto& stage : stage4_3) {
      stage.clear_buffers();
    }
    for (auto& stage : stage2_0) {
      stage.clear_buffers();
    }
    for (auto& stage : stage2_1) {
      stage.clear_buffers();
    }
    for (auto& stage : stage2_2) {
      stage.clear_buffers();
    }
    for (auto& stage : stage2_3) {
      stage.clear_buffers();
    }
  }
};

} // namespace

/**
 * Donwsampler with IIR antialiasing filters.
 */
template<typename Scalar,
         int numCoefsStage0,
         int numCoefsStage1,
         int numCoefsStage2,
         int numCoefsStage3,
         template<int>
         class StageVec8,
         template<int>
         class StageVec4,
         template<int>
         class StageVec2>
class TIirDownsampler final
  : public virtual IirDownsampler<Scalar>
  , public IirOversamplingChain<Scalar,
                                numCoefsStage0,
                                numCoefsStage1,
                                numCoefsStage2,
                                numCoefsStage3,
                                StageVec8,
                                StageVec4,
                                StageVec2>
{
  InterleavedBuffer<Scalar>* output = nullptr;

public:
  /**
   * Constructor.
   * @param designer an IirOversamplerDesigner instance to use to setup the
   * antialiasing filters
   * @param numChannels the number of channels to initialize the downsampler
   * with
   */
  TIirDownsampler(IirOversamplingDesigner const& designer, int numChannels)
    : IirOversamplingChain<Scalar,
                           numCoefsStage0,
                           numCoefsStage1,
                           numCoefsStage2,
                           numCoefsStage3,
                           StageVec8,
                           StageVec4,
                           StageVec2>(designer, numChannels)
  {}

  void ProcessBlock(InterleavedBuffer<Scalar>& input, int numSamples) override
  {
    assert(this->numChannels <= input.GetNumChannels());
    this->PrepareBuffer(numSamples);
    output = &this->buffer[0];
    auto& temp = this->buffer[1];

    switch (this->order) {
      case 0: {
        output = &input;
      } break;
      case 1: {
        this->ApplyStage0(*output, input, numSamples / 2);
      } break;
      case 2: {
        this->ApplyStage1(temp, input, numSamples / 2);
        this->ApplyStage0(*output, temp, numSamples / 4);
      } break;
      case 3: {
        this->ApplyStage2(*output, input, numSamples / 2);
        this->ApplyStage1(temp, *output, numSamples / 4);
        this->ApplyStage0(*output, temp, numSamples / 8);
      } break;
      case 4: {
        this->ApplyStage3(temp, input, numSamples / 2);
        this->ApplyStage2(*output, temp, numSamples / 4);
        this->ApplyStage1(temp, *output, numSamples / 8);
        this->ApplyStage0(*output, temp, numSamples / 16);
      } break;
      default:
        assert(false);
    }
  }

  InterleavedBuffer<Scalar>& GetOutput() override
  {
    assert(output);
    return *output;
  }
};

/**
 * Upsampler with IIR antialiasing filters.
 */
template<typename Scalar,
         int numCoefsStage0,
         int numCoefsStage1,
         int numCoefsStage2,
         int numCoefsStage3,
         template<int>
         class StageVec8,
         template<int>
         class StageVec4,
         template<int>
         class StageVec2>
class TIirUpsampler final
  : public virtual IirUpsampler<Scalar>
  , public IirOversamplingChain<Scalar,
                                numCoefsStage0,
                                numCoefsStage1,
                                numCoefsStage2,
                                numCoefsStage3,
                                StageVec8,
                                StageVec4,
                                StageVec2>
{

public:
  /**
   * Constructor.
   * @param designer an IirOversamplerDesigner instance to use to setup the
   * antialiasing filters
   * @param numChannels the number of channels to initialize the upsampler
   * with
   */
  TIirUpsampler(IirOversamplingDesigner const& designer, int numChannels)
    : IirOversamplingChain<Scalar,
                           numCoefsStage0,
                           numCoefsStage1,
                           numCoefsStage2,
                           numCoefsStage3,
                           StageVec8,
                           StageVec4,
                           StageVec2>(designer, numChannels)
  {}

  void ProcessBlock(Scalar* const* inputs,
                    int numInputSamples,
                    InterleavedBuffer<Scalar>& output) override
  {
    output.SetNumSamples(numInputSamples * this->factor);
    this->PrepareBuffer(numInputSamples);

    auto& input = this->buffer[0];
    auto& temp = this->buffer[1];

    switch (this->order) {
      case 0: {
        output.Interleave(inputs, output.GetNumChannels(), numInputSamples);
      } break;
      case 1: {
        input.Interleave(inputs, output.GetNumChannels(), numInputSamples);
        this->ApplyStage0(output, input, numInputSamples);
      } break;
      case 2: {
        input.Interleave(inputs, output.GetNumChannels(), numInputSamples);
        this->ApplyStage0(temp, input, numInputSamples);
        this->ApplyStage1(output, temp, numInputSamples * 2);
      } break;
      case 3: {
        input.Interleave(inputs, output.GetNumChannels(), numInputSamples);
        this->ApplyStage0(output, input, numInputSamples);
        this->ApplyStage1(temp, output, numInputSamples * 2);
        this->ApplyStage2(output, temp, numInputSamples * 4);
      } break;
      case 4: {
        input.Interleave(inputs, output.GetNumChannels(), numInputSamples);
        this->ApplyStage0(temp, input, numInputSamples);
        this->ApplyStage1(input, temp, numInputSamples * 2);
        this->ApplyStage2(temp, input, numInputSamples * 4);
        this->ApplyStage3(output, temp, numInputSamples * 8);
      } break;
      default:
        assert(false);
    }
  }

  void ProcessBlock(ScalarBuffer<Scalar> const& input,
                    InterleavedBuffer<Scalar>& output) override
  {
    ProcessBlock(input.Get(), input.GetSize(), output);
  }
};

} // namespace oversimple
