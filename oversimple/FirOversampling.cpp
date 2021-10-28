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

#include "oversimple/FirOversampling.hpp"
#include <algorithm>

// macro to send debug messages using Visual Studio
#if (defined _DEBUG)
inline void DEBUG_MESSAGE(char const* message)
{
  OutputDebugStringA(message);
}
#else
#define DEBUG_MESSAGE(x) /*nothing*/
#endif

namespace oversimple::fir {

uint32_t UpSampler::processBlock(ScalarBuffer<double> const& input)
{
  return processBlock(input.get(), input.getNumChannels(), input.getNumSamples());
}

void DownSampler::processBlock(ScalarBuffer<double> const& input,
                               ScalarBuffer<double>& output,
                               uint32_t requiredSamples)
{
  assert(output.getNumChannels() == input.getNumChannels());
  assert(output.getCapacity() >= requiredSamples);
  output.setNumSamples(requiredSamples);
  processBlock(input, output.get(), output.getNumChannels(), requiredSamples);
}

uint32_t UpSampler::processBlock(double* const* input, uint32_t numInputChannels, uint32_t numSamples)
{
  assert(numInputChannels <= numChannels);
  assert(output.getNumChannels() >= numInputChannels);
  assert(output.getCapacity() >= maxOutputLength);
  output.setNumSamples(maxOutputLength);

  uint32_t totalUpSampledSamples = 0;
  for (uint32_t c = 0; c < numInputChannels; ++c) {
    double* outPtr;
    int numInputSamples = (int)numSamples;
    int inputCounter = 0;
    int outputCounter = 0;
    while (numInputSamples > 0) {
      int samplesToProcess = std::min(numInputSamples, (int)fftSamplesPerBlock);
      int numUpSampledSamples = reSamplers[c]->process(&input[c][inputCounter], (int)samplesToProcess, outPtr);
      inputCounter += samplesToProcess;
      numInputSamples -= (int)samplesToProcess;
      if (numUpSampledSamples > 0) {
        auto const totUpSampledSamples = outputCounter + numUpSampledSamples;
        assert(output.getNumSamples() >= totUpSampledSamples);
        std::copy(outPtr, outPtr + numUpSampledSamples, &output[c][outputCounter]);
        outputCounter += numUpSampledSamples;
      }
    }
    totalUpSampledSamples = outputCounter;
  }
  output.setNumSamples(totalUpSampledSamples);
  return totalUpSampledSamples;
}
void DownSampler::processBlock(double* const* input,
                               uint32_t numSamples,
                               double** output,
                               uint32_t numOutputChannels,
                               uint32_t requiredSamples)
{
  assert(numOutputChannels <= numChannels);

  int newBufferCounter = bufferCounter;
  if (numSamples <= fftSamplesPerBlock) {
    for (uint32_t c = 0; c < numOutputChannels; ++c) {
      double* outPtr;
      int const numUpSampledSamples = reSamplers[c]->process(const_cast<double*>(&input[c][0]), numSamples, outPtr);
      int diff = (int)requiredSamples - numUpSampledSamples - bufferCounter;
      if (diff >= 0) {
        std::fill_n(&output[c][0], diff, 0.0);
        std::copy(&buffer[c][0], &buffer[c][0] + bufferCounter, &output[c][diff]);
        std::copy(outPtr, outPtr + numUpSampledSamples, &output[c][diff + bufferCounter]);
        newBufferCounter = 0;
      }
      else { // diff < 0
        int const samplesFromBuffer = std::min(bufferCounter, (int)requiredSamples);
        std::copy(&buffer[c][0], &buffer[c][0] + samplesFromBuffer, &output[c][0]);
        std::copy(&buffer[c][0] + samplesFromBuffer, &buffer[c][0] + bufferCounter, &buffer[c][0]);
        newBufferCounter = bufferCounter - samplesFromBuffer;
        int const wantedSamplesFromReSampler = (int)requiredSamples - samplesFromBuffer;
        int const samplesFromReSampler = std::min(wantedSamplesFromReSampler, numUpSampledSamples);
        std::copy(outPtr, outPtr + samplesFromReSampler, &output[c][samplesFromBuffer]);
        // check buffer size
        auto const neededBufferSize = (uint32_t)(newBufferCounter + numUpSampledSamples - samplesFromReSampler);
        assert(buffer.getCapacity() >= neededBufferSize);
        if (buffer.getNumSamples() < neededBufferSize) {
          buffer.setNumSamples(neededBufferSize);
        }
        // copy tail to buffer
        std::copy(outPtr + samplesFromReSampler, outPtr + numUpSampledSamples, &buffer[c][newBufferCounter]);
      }
    }
    bufferCounter = newBufferCounter;
  }
  else { // numSamples > fftSamplesPerBlock
    for (uint32_t c = 0; c < numOutputChannels; ++c) {
      int inputCounter = 0;
      int outputCounter = 0;
      int numInputSamples = (int)numSamples;
      while (numInputSamples > 0) {
        int samplesToProcess = std::min(numInputSamples, (int)fftSamplesPerBlock);
        double* outPtr;
        int const numUpSampledSamples =
          reSamplers[c]->process(const_cast<double*>(&input[c][inputCounter]), samplesToProcess, outPtr);
        inputCounter += samplesToProcess;
        numInputSamples -= samplesToProcess;
        auto const neededBufferSize = (uint32_t)(bufferCounter + outputCounter + numUpSampledSamples);
        assert(buffer.getCapacity() >= neededBufferSize);
        if (buffer.getNumSamples() < neededBufferSize) {
          buffer.setNumSamples(neededBufferSize);
        }
        if (numUpSampledSamples > 0) {
          std::copy(outPtr, outPtr + numUpSampledSamples, &buffer[c][bufferCounter + outputCounter]);
          outputCounter += numUpSampledSamples;
        }
      }
      newBufferCounter = outputCounter + bufferCounter;
    }
    bufferCounter = newBufferCounter;

    int diff = (int)requiredSamples - bufferCounter;
    if (diff >= 0) {
      for (uint32_t c = 0; c < numOutputChannels; ++c) {
        std::fill_n(&output[c][0], diff, 0.0);
        std::copy(&buffer[c][0], &buffer[c][0] + bufferCounter, &output[c][diff]);
      }
      bufferCounter = 0;
    }
    else { // diff < 0
      for (uint32_t c = 0; c < numOutputChannels; ++c) {
        std::copy(&buffer[c][0], &buffer[c][0] + requiredSamples, &output[c][0]);
        std::copy(&buffer[c][0] + requiredSamples, &buffer[c][0] + bufferCounter, &buffer[c][0]);
      }
    }
  }
}

void DownSampler::processBlock(ScalarBuffer<double> const& input,
                               double** output,
                               uint32_t numOutputChannels,
                               uint32_t requiredSamples)
{
  processBlock(input.get(), input.getNumSamples(), output, numOutputChannels, requiredSamples);
}

void ReSamplerBase::setup()
{
  reSamplers.clear();

  for (uint32_t c = 0; c < numChannels; ++c) {
    reSamplers.push_back(
      std::make_unique<r8b::CDSPResampler24>(1.0, oversamplingRate, fftSamplesPerBlock, transitionBand));
  }
}

ReSamplerBase::ReSamplerBase(uint32_t numChannels,
                             double transitionBand,
                             uint32_t fftSamplesPerBlock,
                             double oversamplingRate)
  : numChannels(numChannels)
  , fftSamplesPerBlock(fftSamplesPerBlock)
  , maxInputLength(fftSamplesPerBlock)
  , transitionBand(transitionBand)
  , oversamplingRate(oversamplingRate)
{}

DownSampler::DownSampler(uint32_t numChannels,
                         double transitionBand,
                         uint32_t fftSamplesPerBlock,
                         double oversamplingRate)
  : ReSamplerBase(numChannels, transitionBand, fftSamplesPerBlock, 1.f / oversamplingRate)
  , maxRequiredOutputLength(fftSamplesPerBlock)
{
  DownSampler::setup();
}

UpSampler::UpSampler(uint32_t numChannels, double transitionBand, uint32_t fftSamplesPerBlock, double oversamplingRate)
  : ReSamplerBase(numChannels, transitionBand, fftSamplesPerBlock, oversamplingRate)
{
  UpSampler::setup();
}

void ReSamplerBase::setNumChannels(uint32_t value)
{
  numChannels = value;
  setup();
}

void ReSamplerBase::setFftSamplesPerBlock(uint32_t value)
{
  fftSamplesPerBlock = value;
  setup();
}

void ReSamplerBase::setTransitionBand(uint32_t value)
{
  transitionBand = value;
  setup();
}

void ReSamplerBase::resetBase()
{
  for (auto& reSampler : reSamplers) {
    reSampler->clear();
  }
}

void ReSamplerBase::prepareBuffersBase(uint32_t numSamples)
{
  maxInputLength = numSamples;
  auto const quot = maxInputLength / fftSamplesPerBlock;
  auto const rem = maxInputLength % fftSamplesPerBlock;
  auto const maxReSamplerOutputLength = (uint32_t)reSamplers[0]->getMaxOutLen((int)fftSamplesPerBlock);
  maxOutputLength = (quot + (rem > 0 ? 1 : 0)) * maxReSamplerOutputLength;
}

void DownSampler::prepareBuffers(uint32_t numInputSamples, uint32_t requiredOutputSamples, bool setAlsoFftBlockSize)
{
  if (setAlsoFftBlockSize) {
    maxInputLength = numInputSamples;
    setFftSamplesPerBlock(numInputSamples);
  }
  else {
    prepareBuffersBase(numInputSamples);
  }
  maxRequiredOutputLength = requiredOutputSamples;
  auto const neededBufferSize = maxOutputLength + std::max(maxOutputLength, requiredOutputSamples);
  if (buffer.getNumSamples() < neededBufferSize) {
    buffer.setNumSamples(neededBufferSize);
  }
}

void UpSampler::setup()
{
  ReSamplerBase::setup();
  output.setNumChannels(numChannels);
  prepareBuffers(maxInputLength);
  reset();
}

void DownSampler::setup()
{
  ReSamplerBase::setup();

  buffer.setNumChannels(numChannels);
  prepareBuffers(maxInputLength, maxRequiredOutputLength);

  reset();
}

uint32_t ReSamplerBase::getNumSamplesBeforeOutputStarts()
{
  if (reSamplers.empty()) {
    assert(false);
    return 0;
  }
  return reSamplers[0]->getInLenBeforeOutStart();
}

void UpSampler::prepareBuffers(uint32_t numSamples, bool setAlsoFftBlockSize)
{
  if (setAlsoFftBlockSize) {
    maxInputLength = numSamples;
    setFftSamplesPerBlock(numSamples);
  }
  else {
    prepareBuffersBase(numSamples);
    output.setNumSamples(maxOutputLength);
  }
}

} // namespace oversimple::fir
