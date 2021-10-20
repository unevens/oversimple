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

#include <utility>

#include "avec/Avec.hpp"

#include "oversimple/IirOversamplingDesigner.hpp"


namespace oversimple::iir::detail {

// implementations

namespace {

template<typename Scalar,
         int numCoefsStage0,
         int numCoefsStage1,
         int numCoefsStage2,
         int numCoefsStage3,
         int numCoefsStage4,
         template<int>
         class StageVec8,
         template<int>
         class StageVec4,
         template<int>
         class StageVec2>
class OversamplingChain
{
protected:
  static constexpr bool VEC8_AVAILABLE = SimdTypes<Scalar>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Scalar>::VEC4_AVAILABLE;
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Scalar>::VEC2_AVAILABLE;

  template<class T>
  using aligned_vector = aligned_vector<T>;

  aligned_vector<StageVec8<numCoefsStage0>> stage8_0;
  aligned_vector<StageVec8<numCoefsStage1>> stage8_1;
  aligned_vector<StageVec8<numCoefsStage2>> stage8_2;
  aligned_vector<StageVec8<numCoefsStage3>> stage8_3;
  aligned_vector<StageVec8<numCoefsStage4>> stage8_4;

  aligned_vector<StageVec4<numCoefsStage0>> stage4_0;
  aligned_vector<StageVec4<numCoefsStage1>> stage4_1;
  aligned_vector<StageVec4<numCoefsStage2>> stage4_2;
  aligned_vector<StageVec4<numCoefsStage3>> stage4_3;
  aligned_vector<StageVec4<numCoefsStage4>> stage4_4;

  aligned_vector<StageVec2<numCoefsStage0>> stage2_0;
  aligned_vector<StageVec2<numCoefsStage1>> stage2_1;
  aligned_vector<StageVec2<numCoefsStage2>> stage2_2;
  aligned_vector<StageVec2<numCoefsStage3>> stage2_3;
  aligned_vector<StageVec2<numCoefsStage4>> stage2_4;

  OversamplingDesigner designer;
  int numChannels;
  int order;
  int maxOrder;
  int factor;
  int maxInputSamples;
  InterleavedBuffer<Scalar> buffer[2];

  std::vector<avec::ScalarBuffer<float>> userScalarBuffers32;
  std::vector<avec::ScalarBuffer<double>> userScalarBuffers64;

  std::vector<avec::InterleavedBuffer<float>> userVecBuffers32;
  std::vector<avec::InterleavedBuffer<double>> userVecBuffers64;

  OversamplingChain(OversamplingDesigner designer_, int numChannels_, int orderToPreallocateFor = 0)
    : designer(std::move(designer_))
    , numChannels(numChannels_)
    , maxInputSamples(256)
    , order(0)
    , maxOrder(orderToPreallocateFor)
    , factor(1)
  {
    assert(designer.getStages().size() == 5);
    setupStages();
  }

  void setupStages()
  {
    int num2, num4, num8;
    avec::getNumOfVecBuffersUsedByInterleavedBuffer<Scalar>(numChannels, num2, num4, num8);
    stage2_0.resize(num2);
    stage2_1.resize(num2);
    stage2_2.resize(num2);
    stage2_3.resize(num2);
    stage2_4.resize(num2);
    stage4_0.resize(num4);
    stage4_1.resize(num4);
    stage4_2.resize(num4);
    stage4_3.resize(num4);
    stage4_4.resize(num4);
    stage8_0.resize(num8);
    stage8_1.resize(num8);
    stage8_2.resize(num8);
    stage8_3.resize(num8);
    stage8_4.resize(num8);

    auto& stages = designer.getStages();
    std::vector<double> coefs;

    stages[0].computeCoefs(coefs);
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
    stages[1].computeCoefs(coefs);
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
    stages[2].computeCoefs(coefs);
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
    stages[3].computeCoefs(coefs);
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
    stages[4].computeCoefs(coefs);
    for (auto& stage : stage4_4) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage8_4) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
    for (auto& stage : stage2_4) {
      stage.set_coefs(&coefs[0]);
      stage.clear_buffers();
    }
  }

  void setupBuffer()
  {
    auto const maxFactor = 1 << maxOrder;
    auto const maxNumOutSamples = maxInputSamples * maxFactor;
    auto const numOutSamples = maxInputSamples * maxFactor;
    for (auto& b : buffer) {
      b.setNumChannels(numChannels);
      b.reserve(maxNumOutSamples);
      b.setNumSamples(numOutSamples);
    }
    for (auto& b : userScalarBuffers32) {
      b.setNumChannels(numChannels);
      b.reserve(maxNumOutSamples);
      b.setNumSamples(numOutSamples);
    }
    for (auto& b : userScalarBuffers64) {
      b.setNumChannels(numChannels);
      b.reserve(maxNumOutSamples);
      b.setNumSamples(numOutSamples);
    }
    for (auto& b : userVecBuffers32) {
      b.setNumChannels(numChannels);
      b.reserve(maxNumOutSamples);
      b.setNumSamples(numOutSamples);
    }
    for (auto& b : userVecBuffers64) {
      b.setNumChannels(numChannels);
      b.reserve(maxNumOutSamples);
      b.setNumSamples(numOutSamples);
    }
  }

  OversamplingDesigner const& getDesigner() const
  {
    return designer;
  }

  void applyStage0(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar> const& input,
                   int numSamples,
                   int numChannelsToProcess)
  {
    if constexpr (VEC2_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage2_0) {
        auto& out = output.getBuffer2(i);
        auto& in = input.getBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 2;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_0) {
        auto& out = output.getBuffer4(i);
        auto& in = input.getBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 4;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_0) {
        auto& out = output.getBuffer8(i);
        auto& in = input.getBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 8;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
  }

  void applyStage1(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar> const& input,
                   int numSamples,
                   int numChannelsToProcess)
  {
    if constexpr (VEC2_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage2_1) {
        auto& out = output.getBuffer2(i);
        auto& in = input.getBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 2;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_1) {
        auto& out = output.getBuffer4(i);
        auto& in = input.getBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 4;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_1) {
        auto& out = output.getBuffer8(i);
        auto& in = input.getBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 8;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
  }

  void applyStage2(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar> const& input,
                   int numSamples,
                   int numChannelsToProcess)
  {
    if constexpr (VEC2_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage2_2) {
        auto& out = output.getBuffer2(i);
        auto& in = input.getBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 2;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_2) {
        auto& out = output.getBuffer4(i);
        auto& in = input.getBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 4;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_2) {
        auto& out = output.getBuffer8(i);
        auto& in = input.getBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 8;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
  }

  void applyStage3(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar> const& input,
                   int numSamples,
                   int numChannelsToProcess)
  {
    if constexpr (VEC2_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage2_3) {
        auto& out = output.getBuffer2(i);
        auto& in = input.getBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 2;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_3) {
        auto& out = output.getBuffer4(i);
        auto& in = input.getBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 4;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_3) {
        auto& out = output.getBuffer8(i);
        auto& in = input.getBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 8;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
  }

  void applyStage4(InterleavedBuffer<Scalar>& output,
                   InterleavedBuffer<Scalar> const& input,
                   int numSamples,
                   int numChannelsToProcess)
  {
    if constexpr (VEC2_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage2_4) {
        auto& out = output.getBuffer2(i);
        auto& in = input.getBuffer2(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 2;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage4_4) {
        auto& out = output.getBuffer4(i);
        auto& in = input.getBuffer4(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 4;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      int i = 0;
      for (auto& stage : stage8_4) {
        auto& out = output.getBuffer8(i);
        auto& in = input.getBuffer8(i);
        stage.process_block(out, in, numSamples);
        ++i;
        numChannelsToProcess -= 8;
        if (numChannelsToProcess <= 0) {
          return;
        }
      }
    }
  }

public:
  int getOrder() const
  {
    return order;
  }
  void setOrder(int value)
  {
    order = value;
    assert(order >= 0 && order <= 5);
    maxOrder = std::max(order, maxOrder);
    factor = 1 << order;
    setupBuffer();
  }
  void prepareBuffer(int maxInputSamples_)
  {
    maxInputSamples = maxInputSamples_;
    setupBuffer();
  }
  void setNumChannels(int value)
  {
    numChannels = value;
    setupBuffer();
    setupStages();
  }
  int getNumChannels() const
  {
    return numChannels;
  }

  void reset()
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
    for (auto& stage : stage8_4) {
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
    for (auto& stage : stage4_4) {
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
    for (auto& stage : stage2_4) {
      stage.clear_buffers();
    }
  }

  void setNumUserScalarBuffers32(int num)
  {
    userScalarBuffers32.resize(num);
    setupBuffer();
  }

  void setNumUserVecBuffers32(int num)
  {
    userVecBuffers32.resize(num);
    setupBuffer();
  }
  void setNumUserScalarBuffers64(int num)
  {
    userScalarBuffers64.resize(num);
    setupBuffer();
  }

  void setNumUserVecBuffers64(int num)
  {
    userVecBuffers64.resize(num);
    setupBuffer();
  }

  avec::ScalarBuffer<float>& getUserScalarBuffer32(int i)
  {
    return userScalarBuffers32[i];
  }

  avec::InterleavedBuffer<float>& getUserVecBuffer32(int i)
  {
    return userVecBuffers32[i];
  }

  avec::ScalarBuffer<double>& getUserScalarBuffer64(int i)
  {
    return userScalarBuffers64[i];
  }

  avec::InterleavedBuffer<double>& getUserVecBuffer64(int i)
  {
    return userVecBuffers64[i];
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
         int numCoefsStage4,
         template<int>
         class StageVec8,
         template<int>
         class StageVec4,
         template<int>
         class StageVec2>
class TDownSampler
  : public OversamplingChain<Scalar,
                             numCoefsStage0,
                             numCoefsStage1,
                             numCoefsStage2,
                             numCoefsStage3,
                             numCoefsStage4,
                             StageVec8,
                             StageVec4,
                             StageVec2>
{
  InterleavedBuffer<Scalar>* output = nullptr;

public:
  /**
   * Constructor.
   * @param designer an ReSamplerDesigner instance to use to setup the
   * antialiasing filters
   * @param numChannels the number of channels to initialize the DownSampler
   * with
   */
  TDownSampler(OversamplingDesigner const& designer, int numChannels, int orderToPreallocateFor = 0)
    : OversamplingChain<Scalar,
                        numCoefsStage0,
                        numCoefsStage1,
                        numCoefsStage2,
                        numCoefsStage3,
                        numCoefsStage4,
                        StageVec8,
                        StageVec4,
                        StageVec2>(designer, numChannels, orderToPreallocateFor)
  {}

  void processBlock(InterleavedBuffer<Scalar> const& input, int numSamples, int numChannelsToProcess)
  {
    if (numChannelsToProcess < 0) {
      numChannelsToProcess = this->numChannels;
    }
    assert(numChannelsToProcess <= this->numChannels);
    assert(numChannelsToProcess <= input.getNumChannels());
    this->prepareBuffer(numSamples);
    output = &this->buffer[0];
    auto& temp = this->buffer[1];

    switch (this->order) {
      case 0: {
        output = const_cast<InterleavedBuffer<Scalar>*>(&input);
      } break;
      case 1: {
        this->applyStage0(*output, input, numSamples / 2, numChannelsToProcess);
      } break;
      case 2: {
        this->applyStage1(temp, input, numSamples / 2, numChannelsToProcess);
        this->applyStage0(*output, temp, numSamples / 4, numChannelsToProcess);
      } break;
      case 3: {
        this->applyStage2(*output, input, numSamples / 2, numChannelsToProcess);
        this->applyStage1(temp, *output, numSamples / 4, numChannelsToProcess);
        this->applyStage0(*output, temp, numSamples / 8, numChannelsToProcess);
      } break;
      case 4: {
        this->applyStage3(temp, input, numSamples / 2, numChannelsToProcess);
        this->applyStage2(*output, temp, numSamples / 4, numChannelsToProcess);
        this->applyStage1(temp, *output, numSamples / 8, numChannelsToProcess);
        this->applyStage0(*output, temp, numSamples / 16, numChannelsToProcess);
      } break;
      case 5: {
        this->applyStage4(*output, input, numSamples / 2, numChannelsToProcess);
        this->applyStage3(temp, *output, numSamples / 4, numChannelsToProcess);
        this->applyStage2(*output, temp, numSamples / 8, numChannelsToProcess);
        this->applyStage1(temp, *output, numSamples / 16, numChannelsToProcess);
        this->applyStage0(*output, temp, numSamples / 32, numChannelsToProcess);
      } break;
      default:
        assert(false);
    }
  }

  InterleavedBuffer<Scalar>& getOutput()
  {
    assert(output);
    return *output;
  }
};

/**
 * UpSampler with IIR antialiasing filters.
 */
template<typename Scalar,
         int numCoefsStage0,
         int numCoefsStage1,
         int numCoefsStage2,
         int numCoefsStage3,
         int numCoefsStage4,
         template<int>
         class StageVec8,
         template<int>
         class StageVec4,
         template<int>
         class StageVec2>
class TUpSampler
  : public OversamplingChain<Scalar,
                             numCoefsStage0,
                             numCoefsStage1,
                             numCoefsStage2,
                             numCoefsStage3,
                             numCoefsStage4,
                             StageVec8,
                             StageVec4,
                             StageVec2>
{
  using Chain = OversamplingChain<Scalar,
                                  numCoefsStage0,
                                  numCoefsStage1,
                                  numCoefsStage2,
                                  numCoefsStage3,
                                  numCoefsStage4,
                                  StageVec8,
                                  StageVec4,
                                  StageVec2>;

public:
  /**
   * Constructor.
   * @param designer an ReSamplerDesigner instance to use to setup the
   * antialiasing filters
   * @param numChannels the number of channels to initialize the UpSampler
   * with
   */
  TUpSampler(OversamplingDesigner const& designer, int numChannels, int orderToPreallocateFor)
    : Chain(designer, numChannels, orderToPreallocateFor)
  {}
  /**
   * Up-samples an already interleaved input.
   * @param input an InterleavedBuffer<Scalar> holding the input samples
   * @param numInputSamples the number of samples in each channel of the input
   * buffer
   * @param output an InterleavedBuffer to hold the up-sampled samples
   * @param numChannelsToProcess the number of channels to process. If negative,
   * all channels will be processed.
   */
  void processBlock(InterleavedBuffer<Scalar> const& input,
                    int numInputSamples,
                    InterleavedBuffer<Scalar>& output,
                    int numChannelsToProcess)
  {
    if (numChannelsToProcess < 0) {
      numChannelsToProcess = this->numChannels;
    }
    assert(numChannelsToProcess <= this->numChannels);
    assert(numChannelsToProcess <= input.getNumChannels());
    assert(numChannelsToProcess <= output.getNumChannels());
    output.setNumSamples(numInputSamples * this->factor);
    this->prepareBuffer(numInputSamples);

    auto& temp2 = this->buffer[0];
    auto& temp = this->buffer[1];

    switch (this->order) {
      case 0: {
        output.copyFrom(input, numInputSamples, numChannelsToProcess);
      } break;
      case 1: {
        this->applyStage0(output, input, numInputSamples, numChannelsToProcess);
      } break;
      case 2: {
        this->applyStage0(temp, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(output, temp, numInputSamples * 2, numChannelsToProcess);
      } break;
      case 3: {
        this->applyStage0(output, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(temp, output, numInputSamples * 2, numChannelsToProcess);
        this->applyStage2(output, temp, numInputSamples * 4, numChannelsToProcess);
      } break;
      case 4: {
        this->applyStage0(temp, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(temp2, temp, numInputSamples * 2, numChannelsToProcess);
        this->applyStage2(temp, temp2, numInputSamples * 4, numChannelsToProcess);
        this->applyStage3(output, temp, numInputSamples * 8, numChannelsToProcess);
      } break;
      case 5: {
        this->applyStage0(output, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(temp, output, numInputSamples * 2, numChannelsToProcess);
        this->applyStage2(output, temp, numInputSamples * 4, numChannelsToProcess);
        this->applyStage3(temp, output, numInputSamples * 8, numChannelsToProcess);
        this->applyStage4(output, temp, numInputSamples * 16, numChannelsToProcess);
      } break;
      default:
        assert(false);
    }
  }
  /**
   * Up-samples the input.
   * @param input a pointer to the memory holding the input samples
   * @param numInputSamples the number of samples in each channel of the input
   * buffer
   * @param output an InterleavedBuffer to hold the up-sampled samples
   * @param numChannelsToProcess the number of channels to process. If negative,
   * all channels will be processed.
   */
  void processBlock(Scalar* const* inputs,
                    int numInputSamples,
                    InterleavedBuffer<Scalar>& output,
                    int numChannelsToProcess)
  {
    if (numChannelsToProcess < 0) {
      numChannelsToProcess = this->numChannels;
    }
    assert(numChannelsToProcess <= this->numChannels);
    assert(numChannelsToProcess <= output.getNumChannels());
    output.setNumSamples(numInputSamples * this->factor);
    this->prepareBuffer(numInputSamples);

    auto& input = this->buffer[0];
    auto& temp = this->buffer[1];

    switch (this->order) {
      case 0: {
        output.interleave(inputs, output.getNumChannels(), numInputSamples);
      } break;
      case 1: {
        input.interleave(inputs, output.getNumChannels(), numInputSamples);
        this->applyStage0(output, input, numInputSamples, numChannelsToProcess);
      } break;
      case 2: {
        input.interleave(inputs, output.getNumChannels(), numInputSamples);
        this->applyStage0(temp, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(output, temp, numInputSamples * 2, numChannelsToProcess);
      } break;
      case 3: {
        input.interleave(inputs, output.getNumChannels(), numInputSamples);
        this->applyStage0(output, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(temp, output, numInputSamples * 2, numChannelsToProcess);
        this->applyStage2(output, temp, numInputSamples * 4, numChannelsToProcess);
      } break;
      case 4: {
        input.interleave(inputs, output.getNumChannels(), numInputSamples);
        this->applyStage0(temp, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(input, temp, numInputSamples * 2, numChannelsToProcess);
        this->applyStage2(temp, input, numInputSamples * 4, numChannelsToProcess);
        this->applyStage3(output, temp, numInputSamples * 8, numChannelsToProcess);
      } break;
      case 5: {
        input.interleave(inputs, output.getNumChannels(), numInputSamples);
        this->applyStage0(output, input, numInputSamples, numChannelsToProcess);
        this->applyStage1(temp, output, numInputSamples * 2, numChannelsToProcess);
        this->applyStage2(output, temp, numInputSamples * 4, numChannelsToProcess);
        this->applyStage3(temp, output, numInputSamples * 8, numChannelsToProcess);
        this->applyStage4(output, temp, numInputSamples * 16, numChannelsToProcess);
      } break;
      default:
        assert(false);
    }
  }
  /**
   * Up-samples the input.
   * @param input a ScalarBuffer that holds the input samples
   * @param output an InterleavedBuffer to hold the up-sampled samples
   * @param numChannelsToProcess the number of channels to process. If negative,
   * all channels will be processed.
   */
  void processBlock(ScalarBuffer<Scalar> const& input, InterleavedBuffer<Scalar>& output, int numChannelsToProcess)
  {
    processBlock(input.get(), input.getNumSamples(), output, numChannelsToProcess);
  }
};

} // namespace oversimple::iir::detail
