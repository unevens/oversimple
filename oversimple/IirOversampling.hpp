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
#include "oversimple/Hiir.hpp"

namespace oversimple::iir {

namespace detail {

namespace {
/**
 * A class implementing common functionality for the IIR resamplers.
 */
template<typename Number,
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
  static constexpr bool VEC8_AVAILABLE = SimdTypes<Number>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Number>::VEC4_AVAILABLE;
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Number>::VEC2_AVAILABLE;

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
  uint32_t numChannels;
  uint32_t order;
  uint32_t maxOrder;
  uint32_t maxInputSamples;
  InterleavedBuffer<Number> buffer[2];

  OversamplingChain(OversamplingDesigner designer_, uint32_t numChannels_, uint32_t orderToPreallocateFor = 1)
    : designer(std::move(designer_))
    , numChannels(numChannels_)
    , maxInputSamples(256)
    , order(1)
    , maxOrder(orderToPreallocateFor)
  {
    assert(designer.getStages().size() == 5);
    setupStages();
  }

  void setupStages()
  {
    uint32_t num2, num4, num8;
    avec::getNumOfVecBuffersUsedByInterleavedBuffer<Number>(numChannels, num2, num4, num8);
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
    auto const factor = 1 << order;
    auto const maxNumOutSamples = maxInputSamples * maxFactor;
    auto const numOutSamples = maxInputSamples * factor;
    for (auto& b : buffer) {
      b.setNumChannels(numChannels);
      b.reserve(maxNumOutSamples);
      b.setNumSamples(numOutSamples);
    }
  }

  OversamplingDesigner const& getDesigner() const
  {
    return designer;
  }

  void applyStage0(InterleavedBuffer<Number>& output, InterleavedBuffer<Number> const& input, uint32_t numSamples)
  {
    auto numChannelsToProcess = numChannels;
    if constexpr (VEC2_AVAILABLE) {
      uint32_t i = 0;
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
      uint32_t i = 0;
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
      uint32_t i = 0;
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

  void applyStage1(InterleavedBuffer<Number>& output, InterleavedBuffer<Number> const& input, uint32_t numSamples)
  {
    auto numChannelsToProcess = numChannels;
    if constexpr (VEC2_AVAILABLE) {
      uint32_t i = 0;
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
      uint32_t i = 0;
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
      uint32_t i = 0;
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

  void applyStage2(InterleavedBuffer<Number>& output, InterleavedBuffer<Number> const& input, uint32_t numSamples)
  {
    auto numChannelsToProcess = numChannels;
    if constexpr (VEC2_AVAILABLE) {
      uint32_t i = 0;
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
      uint32_t i = 0;
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
      uint32_t i = 0;
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

  void applyStage3(InterleavedBuffer<Number>& output, InterleavedBuffer<Number> const& input, uint32_t numSamples)
  {
    auto numChannelsToProcess = numChannels;
    if constexpr (VEC2_AVAILABLE) {
      uint32_t i = 0;
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
      uint32_t i = 0;
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
      uint32_t i = 0;
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

  void applyStage4(InterleavedBuffer<Number>& output, InterleavedBuffer<Number> const& input, uint32_t numSamples)
  {
    auto numChannelsToProcess = numChannels;
    if constexpr (VEC2_AVAILABLE) {
      uint32_t i = 0;
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
      uint32_t i = 0;
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
      uint32_t i = 0;
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
  /**
   * @return the order of oversampling currently in use
   */
  uint32_t getOrder() const
  {
    return order;
  }

  /**
   * Sets the order of oversampling to be used. It must be less or equal to the maximum order set
   * @value the order to set
   * @return true if the order was set correctly, false otherwise
   */
  bool setOrder(uint32_t value)
  {
    if (value < 1 || value > 5) {
      return false;
    }
    order = value;
    return true;
  }

  /**
   * Sets the maximum order of oversampling that can be used. Allocates internal buffers accordingly.
   * @value the maximum order to set
   * @return true if the order was set correctly, false otherwise
   */
  bool setMaxOrder(uint32_t value)
  {
    if (value < 1 || value > 5) {
      return false;
    }
    maxOrder = value;
    setupBuffer();
    return true;
  }

  /**
   * Prepares the internal buffers to receive the specified amount of samples.
   * @maxInputSamples_ the maximum amount of samples that can be processed by a single processing call
   */
  void prepareBuffer(uint32_t maxInputSamples_)
  {
    maxInputSamples = maxInputSamples_;
    setupBuffer();
  }

  /**
   * Sets the number of channels that the resampler can work with
   * @value the number of channels the resampler will be able to work with
   */
  void setNumChannels(uint32_t value)
  {
    numChannels = value;
    setupBuffer();
    setupStages();
  }

  /**
   * @return the number of channels the resampler is able to work with
   */
  uint32_t getNumChannels() const
  {
    return numChannels;
  }

  /**
   * Resets the state of the antialiasing filters
   */
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
};

} // namespace

/**
 * DownSampler with IIR antialiasing filters.
 */
template<typename Number,
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
  : public OversamplingChain<Number,
                             numCoefsStage0,
                             numCoefsStage1,
                             numCoefsStage2,
                             numCoefsStage3,
                             numCoefsStage4,
                             StageVec8,
                             StageVec4,
                             StageVec2>
{
public:
  /**
   * Constructor.
   * @param designer an ReSamplerDesigner instance to use to setup the
   * antialiasing filters
   * @param numChannels the number of channels to initialize the DownSampler
   * with
   * @param orderToPreallocateFor the maximum order of oversampling for which to allocate resources for
   */
  TDownSampler(OversamplingDesigner const& designer, uint32_t numChannels, uint32_t orderToPreallocateFor = 1)
    : OversamplingChain<Number,
                        numCoefsStage0,
                        numCoefsStage1,
                        numCoefsStage2,
                        numCoefsStage3,
                        numCoefsStage4,
                        StageVec8,
                        StageVec4,
                        StageVec2>(designer, numChannels, orderToPreallocateFor)
  {}

  /**
   * Down-samples the input.
   * @param input an InterleavedBuffer holding the input
   */
  void processBlock(InterleavedBuffer<Number> const& input)
  {
    assert(this->numChannels == input.getNumChannels());
    auto const numSamples = input.getNumSamples();
    assert(numSamples <= this->maxInputSamples);

    auto& output = this->buffer[0];
    auto& temp = this->buffer[1];

    switch (this->order) {
      case 1: {
        this->applyStage0(output, input, numSamples / 2);
      } break;
      case 2: {
        this->applyStage1(temp, input, numSamples / 2);
        this->applyStage0(output, temp, numSamples / 4);
      } break;
      case 3: {
        this->applyStage2(output, input, numSamples / 2);
        this->applyStage1(temp, output, numSamples / 4);
        this->applyStage0(output, temp, numSamples / 8);
      } break;
      case 4: {
        this->applyStage3(temp, input, numSamples / 2);
        this->applyStage2(output, temp, numSamples / 4);
        this->applyStage1(temp, output, numSamples / 8);
        this->applyStage0(output, temp, numSamples / 16);
      } break;
      case 5: {
        this->applyStage4(output, input, numSamples / 2);
        this->applyStage3(temp, output, numSamples / 4);
        this->applyStage2(output, temp, numSamples / 8);
        this->applyStage1(temp, output, numSamples / 16);
        this->applyStage0(output, temp, numSamples / 32);
      } break;
      default:
        assert(false);
    }
  }

  /**
   * @return an InterleavedBuffer holding the output
   */
  InterleavedBuffer<Number>& getOutput()
  {
    return this->buffer[0];
  }
};

/**
 * UpSampler with IIR antialiasing filters.
 */
template<typename Number,
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
  : public OversamplingChain<Number,
                             numCoefsStage0,
                             numCoefsStage1,
                             numCoefsStage2,
                             numCoefsStage3,
                             numCoefsStage4,
                             StageVec8,
                             StageVec4,
                             StageVec2>
{
  using Chain = OversamplingChain<Number,
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
   * @param orderToPreallocateFor the maximum order of oversampling for which to allocate resources for
   */
  TUpSampler(OversamplingDesigner const& designer, uint32_t numChannels, uint32_t orderToPreallocateFor)
    : Chain(designer, numChannels, orderToPreallocateFor)
  {}
  /**
   * Up-samples an already interleaved input.
   * @param input an InterleavedBuffer<Number> holding the input samples
   */
  void processBlock(InterleavedBuffer<Number> const& input)
  {
    auto const numInputSamples = input.getNumSamples();

    assert(input.getNumChannels() == this->numChannels);
    assert(numInputSamples <= this->maxInputSamples);

    auto& output = this->buffer[1];
    auto& temp = this->buffer[0];

    switch (this->order) {
      case 1: {
        this->applyStage0(output, input, numInputSamples);
      } break;
      case 2: {
        this->applyStage0(temp, input, numInputSamples);
        this->applyStage1(output, temp, numInputSamples * 2);
      } break;
      case 3: {
        this->applyStage0(output, input, numInputSamples);
        this->applyStage1(temp, output, numInputSamples * 2);
        this->applyStage2(output, temp, numInputSamples * 4);
      } break;
      case 4: {
        this->applyStage0(temp, input, numInputSamples);
        this->applyStage1(output, temp, numInputSamples * 2);
        this->applyStage2(temp, output, numInputSamples * 4);
        this->applyStage3(output, temp, numInputSamples * 8);
      } break;
      case 5: {
        this->applyStage0(output, input, numInputSamples);
        this->applyStage1(temp, output, numInputSamples * 2);
        this->applyStage2(output, temp, numInputSamples * 4);
        this->applyStage3(temp, output, numInputSamples * 8);
        this->applyStage4(output, temp, numInputSamples * 16);
      } break;
      default:
        assert(false);
    }
  }

  /**
   * Up-samples the input.
   * @param input a pointer to the memory holding the input samples.
   * @param numInputSamples the number of samples in each channel of the input
   * buffer
   */
  void processBlock(Number* const* inputs, uint32_t numInputSamples)
  {
    assert(numInputSamples <= this->maxInputSamples);

    switch (this->order) {
      case 1: {
        this->buffer[0].interleave(inputs, this->numChannels, numInputSamples);
        this->applyStage0(this->buffer[1], this->buffer[0], numInputSamples);
      } break;
      case 2: {
        this->buffer[1].interleave(inputs, this->numChannels, numInputSamples);
        this->applyStage0(this->buffer[0], this->buffer[1], numInputSamples);
        this->applyStage1(this->buffer[1], this->buffer[0], numInputSamples * 2);
      } break;
      case 3: {
        this->buffer[0].interleave(inputs, this->numChannels, numInputSamples);
        this->applyStage0(this->buffer[1], this->buffer[0], numInputSamples);
        this->applyStage1(this->buffer[0], this->buffer[1], numInputSamples * 2);
        this->applyStage2(this->buffer[1], this->buffer[0], numInputSamples * 4);
      } break;
      case 4: {
        this->buffer[1].interleave(inputs, this->numChannels, numInputSamples);
        this->applyStage0(this->buffer[0], this->buffer[1], numInputSamples);
        this->applyStage1(this->buffer[1], this->buffer[0], numInputSamples * 2);
        this->applyStage2(this->buffer[0], this->buffer[1], numInputSamples * 4);
        this->applyStage3(this->buffer[1], this->buffer[0], numInputSamples * 8);
      } break;
      case 5: {
        this->buffer[0].interleave(inputs, this->numChannels, numInputSamples);
        this->applyStage0(this->buffer[1], this->buffer[0], numInputSamples);
        this->applyStage1(this->buffer[0], this->buffer[1], numInputSamples * 2);
        this->applyStage2(this->buffer[1], this->buffer[0], numInputSamples * 4);
        this->applyStage3(this->buffer[0], this->buffer[1], numInputSamples * 8);
        this->applyStage4(this->buffer[1], this->buffer[0], numInputSamples * 16);
      } break;
      default:
        assert(false);
    }
  }
  /**
   * Up-samples the input.
   * @param input a Buffer that holds the input samples
   */
  void processBlock(Buffer<Number> const& input)
  {
    assert(input.getNumChannels() == this->numChannels);
    processBlock(input.get(), input.getNumSamples());
  }

  /**
   * @return an InterleavedBuffer holding the output
   */
  InterleavedBuffer<Number>& getOutput()
  {
    return this->buffer[1];
  }
};

/**
 * Static class used to deduce the right SIMD implementation for the current architecture
 */
template<typename Number>
class UpSamplerFactory final
{
  template<int NC>
  using FakeUpSamplerStage8Double = hiir::FakeInterface;
  template<int NC>
  using FakeUpSamplerStage2Float = hiir::FakeInterface;

#if AVEC_X86

public:
  template<int NC>
  using Stage8 = typename std::
    conditional<std::is_same<Number, float>::value, hiir::Upsampler2x8Avx<NC>, FakeUpSamplerStage8Double<NC>>::type;
  template<int NC>
  using Stage4 = typename std::
    conditional<std::is_same<Number, float>::value, hiir::Upsampler2x4Sse<NC>, hiir::Upsampler2x4F64Avx<NC>>::type;
  template<int NC>
  using Stage2 = typename std::
    conditional<std::is_same<Number, float>::value, FakeUpSamplerStage2Float<NC>, hiir::Upsampler2x2F64Sse2<NC>>::type;

#elif AVEC_ARM

private:
  template<int NC>
  using FakeUpSamplerStage4Double = hiir::Upsampler2x4F64Avx<NC>;

public:
  template<int NC>
  using Stage8 = FakeUpSamplerStage8Double<NC>;
  template<int NC>
  using Stage4 = typename std::
    conditional<std::is_same<Number, float>::value, hiir::Upsampler2x4Neon<NC>, FakeUpSamplerStage4Double<NC>>::type;
  template<int NC>
  using Stage2 = typename std::
    conditional<std::is_same<Number, float>::value, FakeUpSamplerStage2Float<NC>, hiir::Upsampler2x2F64Neon<NC>>::type;

#else

  static_assert(false, "Unknown platform");

#endif

public:
  using UpSampler = TUpSampler<Number, 11, 5, 3, 3, 2, Stage8, Stage4, Stage2>;
};

/**
 * Static class used to deduce the right SIMD implementation for the current architecture
 */
template<typename Number>
class DownSamplerFactory final
{
  template<int NC>
  using FakeDownsamplerStage8Double = hiir::Downsampler2x4F64Avx<NC>;
  template<int NC>
  using FakeDownsamplerStage2Float = hiir::Downsampler2x4Sse<NC>;

#if AVEC_X86

public:
  template<int NC>
  using Stage8 = typename std::
    conditional<std::is_same<Number, float>::value, hiir::Downsampler2x8Avx<NC>, FakeDownsamplerStage8Double<NC>>::type;

  template<int NC>
  using Stage4 = typename std::
    conditional<std::is_same<Number, float>::value, hiir::Downsampler2x4Sse<NC>, hiir::Downsampler2x4F64Avx<NC>>::type;

  template<int NC>
  using Stage2 = typename std::conditional<std::is_same<Number, float>::value,
                                           FakeDownsamplerStage2Float<NC>,
                                           hiir::Downsampler2x2F64Sse2<NC>>::type;

#elif AVEC_ARM

private:
  template<int NC>
  using FakeDownsamplerStage4Double = hiir::Downsampler2x4F64Avx<NC>;

public:
  template<int NC>
  using Stage8 = FakeDownsamplerStage8Double<NC>;
  template<int NC>
  using Stage4 = typename std::conditional<std::is_same<Number, float>::value,
                                           hiir::Downsampler2x4Neon<NC>,
                                           FakeDownsamplerStage4Double<NC>>::type;
  template<int NC>
  using Stage2 = typename std::conditional<std::is_same<Number, float>::value,
                                           FakeDownsamplerStage2Float<NC>,
                                           hiir::Downsampler2x2F64Neon<NC>>::type;

#else

  static_assert(false, "Unknown platform");

#endif

public:
  using DownSampler = TDownSampler<Number, 11, 5, 3, 3, 2, Stage8, Stage4, Stage2>;
};

} // namespace detail

/**
 * DownSampler with IIR antialiasing filters.
 * The filters are setup to achieve 140dB of attenuation and a transition band of 0.0443
 */
template<typename Number>
class DownSampler final : public detail::DownSamplerFactory<Number>::DownSampler
{
public:
  explicit DownSampler(uint32_t numChannels, uint32_t orderToPreallocateFor = 1)
    : detail::DownSamplerFactory<Number>::DownSampler(detail::getOversamplingPreset(0),
                                                      numChannels,
                                                      orderToPreallocateFor)
  {}
};

/**
 * UpSampler with IIR antialiasing filters.
 * The filters are setup to achieve 140dB of attenuation and a transition band of 0.0443
 */
template<typename Number>
class UpSampler final : public detail::UpSamplerFactory<Number>::UpSampler
{
public:
  explicit UpSampler(uint32_t numChannels, uint32_t orderToPreallocateFor = 1)
    : detail::UpSamplerFactory<Number>::UpSampler(detail::getOversamplingPreset(0), numChannels, orderToPreallocateFor)
  {}
};

} // namespace oversimple::iir
