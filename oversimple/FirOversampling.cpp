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

int UpSampler::processBlock(ScalarBuffer<double> const& input)
{
  return processBlock(input.get(), input.getNumChannels(), input.getNumSamples());
}

void DownSampler::processBlock(ScalarBuffer<double> const& input, ScalarBuffer<double>& output, int requiredSamples)
{
  output.setNumChannelsAndSamples(input.getNumChannels(), input.getNumSamples());
  processBlock(input, output.get(), output.getNumChannels(), requiredSamples);
}

int UpSampler::processBlock(double* const* input, int numInputChannels, int numSamples)
{
  assert(numInputChannels <= numChannels);
  int numOutputSamples = (int)std::ceil(numSamples * oversamplingRate);
  if (output.getNumChannels() < numInputChannels || output.getCapacity() < numOutputSamples) {
    DEBUG_MESSAGE("A UpSampler object had to allocate memory! Has "
                  "prepareBuffers been called?\n");
  }
  output.setNumChannelsAndSamples(numInputChannels, numOutputSamples);

  if (oversamplingRate == 1) {
    for (int c = 0; c < numInputChannels; ++c) {
      std::copy(&input[c][0], &input[c][0] + numSamples, &output[c][0]);
    }
    return numSamples;
  }
  int totalUpSampledSamples = 0;
  for (int c = 0; c < numInputChannels; ++c) {
    double* outPtr;
    int numInputSamples = numSamples;
    int inputCounter = 0;
    int outputCounter = 0;
    while (numInputSamples > 0) {
      int samplesToProcess = std::min(numInputSamples, maxSamplesPerBlock);
      int numUpSampledSamples = reSamplers[c]->process(&input[c][inputCounter], samplesToProcess, outPtr);
      inputCounter += samplesToProcess;
      numInputSamples -= samplesToProcess;
      if (numUpSampledSamples > 0) {
        if (outputCounter + numUpSampledSamples > numOutputSamples) {
          DEBUG_MESSAGE("A UpSampler object had to allocate memory due to a "
                        "fluctuation, this shold not happen!\n");
          output.setNumSamples(outputCounter + numOutputSamples);
        }
        std::copy(outPtr, outPtr + numUpSampledSamples, &output[c][outputCounter]);
        outputCounter += numUpSampledSamples;
      }
    }
    totalUpSampledSamples = outputCounter;
  }
  return totalUpSampledSamples;
}
void DownSampler::processBlock(double* const* input,
                               int numSamples,
                               double** output,
                               int numOutputChannels,
                               int requiredSamples)
{
  assert(numOutputChannels <= numChannels);

  if (oversamplingRate == 1.0) {
    for (int c = 0; c < numOutputChannels; ++c) {
      std::copy(&input[c][0], &input[c][0] + numSamples, &output[c][0]);
    }
    return;
  }

  int newBufferCounter = bufferCounter;
  if (numSamples <= maxSamplesPerBlock) {
    for (int c = 0; c < numOutputChannels; ++c) {
      double* outPtr;
      int numUpsampledSamples = reSamplers[c]->process(const_cast<double*>(&input[c][0]), numSamples, outPtr);
      int diff = requiredSamples - numUpsampledSamples - bufferCounter;
      if (diff >= 0) {
        std::fill_n(&output[c][0], diff, 0.0);
        std::copy(&buffer[c][0], &buffer[c][0] + bufferCounter, &output[c][diff]);
        std::copy(outPtr, outPtr + numUpsampledSamples, &output[c][diff + bufferCounter]);
        newBufferCounter = 0;
      }
      else { // diff < 0
        int samplesFromBuffer = std::min(bufferCounter, requiredSamples);
        std::copy(&buffer[c][0], &buffer[c][0] + samplesFromBuffer, &output[c][0]);
        std::copy(&buffer[c][0] + samplesFromBuffer, &buffer[c][0] + bufferCounter, &buffer[c][0]);
        newBufferCounter = bufferCounter - samplesFromBuffer;
        int wantedSamplesFromReSampler = requiredSamples - samplesFromBuffer;
        int samplesFromReSampler = std::min(wantedSamplesFromReSampler, numUpsampledSamples);
        std::copy(outPtr, outPtr + samplesFromReSampler, &output[c][samplesFromBuffer]);
        // check buffer size
        int neededBufferSize = newBufferCounter + numUpsampledSamples - samplesFromReSampler;
        if (buffer.getNumSamples() < neededBufferSize) {
          if (buffer.getCapacity() < neededBufferSize) {
            DEBUG_MESSAGE("A DownSampler object had to allocate "
                          "memory! Has prepareBuffers been called?\n");
          }
          buffer.setNumSamples(neededBufferSize);
        }
        // copy tail to buffer
        std::copy(outPtr + samplesFromReSampler, outPtr + numUpsampledSamples, &buffer[c][newBufferCounter]);
      }
    }
    bufferCounter = newBufferCounter;
  }
  else { // numSamples > maxSamplesPerBlock
    for (int c = 0; c < numOutputChannels; ++c) {
      int inputCounter = 0;
      int outputCounter = 0;
      int numInputSamples = numSamples;
      while (numInputSamples > 0) {
        int samplesToProcess = std::min(numInputSamples, maxSamplesPerBlock);
        double* outPtr;
        int numUpSampledSamples =
          reSamplers[c]->process(const_cast<double*>(&input[c][inputCounter]), samplesToProcess, outPtr);
        inputCounter += samplesToProcess;
        numInputSamples -= samplesToProcess;
        int neededBufferSize = bufferCounter + outputCounter + numUpSampledSamples;
        if (buffer.getNumSamples() < neededBufferSize) {
          if (buffer.getCapacity() < neededBufferSize) {
            DEBUG_MESSAGE("A DownSampler object had to allocate "
                          "memory! Has prepareBuffers been called?");
          }
          buffer.setNumSamples(bufferCounter + outputCounter + numUpSampledSamples);
        }
        if (numUpSampledSamples > 0) {
          std::copy(outPtr, outPtr + numUpSampledSamples, &buffer[c][bufferCounter + outputCounter]);
          outputCounter += numUpSampledSamples;
        }
      }
      newBufferCounter = outputCounter + bufferCounter;
    }
    bufferCounter = newBufferCounter;

    int diff = requiredSamples - bufferCounter;
    if (diff >= 0) {
      for (int c = 0; c < numOutputChannels; ++c) {
        std::fill_n(&output[c][0], diff, 0.0);
        std::copy(&buffer[c][0], &buffer[c][0] + bufferCounter, &output[c][diff]);
      }
      bufferCounter = 0;
    }
    else { // diff < 0
      for (int c = 0; c < numOutputChannels; ++c) {
        std::copy(&buffer[c][0], &buffer[c][0] + requiredSamples, &output[c][0]);
        std::copy(&buffer[c][0] + requiredSamples, &buffer[c][0] + bufferCounter, &buffer[c][0]);
      }
    }
  }
}

void DownSampler::processBlock(ScalarBuffer<double> const& input,
                               double** output,
                               int numOutputChannels,
                               int requiredSamples)
{
  processBlock(input.get(), input.getNumSamples(), output, numOutputChannels, requiredSamples);
}

void ReSamplerBase::setup()
{
  reSamplers.clear();

  for (int c = 0; c < numChannels; ++c) {
    reSamplers.push_back(
      std::make_unique<r8b::CDSPResampler24>(1.0, oversamplingRate, maxSamplesPerBlock, transitionBand));
  }
}

ReSamplerBase::ReSamplerBase(int numChannels, double transitionBand, int maxSamplesPerBlock, double oversamplingRate)
  : numChannels(numChannels)
  , maxSamplesPerBlock(maxSamplesPerBlock)
  , maxInputLength(maxSamplesPerBlock)
  , transitionBand(transitionBand)
  , oversamplingRate(oversamplingRate)
{}

DownSampler::DownSampler(int numChannels, double transitionBand, int maxSamplesPerBlock, double oversamplingRate)
  : ReSamplerBase(numChannels, transitionBand, maxSamplesPerBlock, 1.f / oversamplingRate)
  , maxRequiredOutputLength(maxSamplesPerBlock)
{
  DownSampler::setup();
}

UpSampler::UpSampler(int numChannels, double transitionBand, int maxSamplesPerBlock, double oversamplingRate)
  : ReSamplerBase(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
{
  UpSampler::setup();
}

void ReSamplerBase::setNumChannels(int value)
{
  numChannels = value;
  setup();
}

void ReSamplerBase::setMaxSamplesPerBlock(int value)
{
  maxSamplesPerBlock = value;
  setup();
}

void ReSamplerBase::setTransitionBand(int value)
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

void ReSamplerBase::prepareBuffersBase(int numSamples)
{
  maxInputLength = numSamples;
  auto d = std::div(maxInputLength, maxSamplesPerBlock);
  int maxReSamplerOutputLength = reSamplers[0]->getMaxOutLen(maxSamplesPerBlock);
  maxOutputLength = (d.quot + (d.rem > 0 ? 1 : 0)) * maxReSamplerOutputLength;
}

void DownSampler::prepareBuffers(int numInputSamples, int requiredOutputSamples)
{
  prepareBuffersBase(numInputSamples);
  maxRequiredOutputLength = requiredOutputSamples;
  int neededBufferSize = maxOutputLength + std::max(maxOutputLength, requiredOutputSamples);
  // maybe buffer.setNumSamples(maxOutputLength); is enough?
  if (buffer.getNumSamples() < neededBufferSize) {
    buffer.setNumSamples(neededBufferSize);
  }
}

void UpSampler::setup()
{
  ReSamplerBase::setup();

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

int ReSamplerBase::getNumSamplesBeforeOutputStarts()
{
  if (reSamplers.empty()) {
    DEBUG_MESSAGE("Asking the number of samples before the output of the "
                  "reSamplers starts when there are 0 allocated reSamplers.");
    return 0;
  }
  return reSamplers[0]->getInLenBeforeOutStart();
}

void UpSampler::prepareBuffers(int numSamples)
{
  prepareBuffersBase(numSamples);
}

} // namespace oversimple::fir
