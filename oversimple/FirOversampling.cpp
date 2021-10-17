/*
Copyright 2019-2020 Dario Mambro

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

namespace oversimple {

int FirUnbufferedReSampler::processBlock(ScalarBuffer<double> const& input, ScalarBuffer<double>& output)
{
  return processBlock(input.get(), input.getNumChannels(), input.getNumSamples(), output);
}

void FirBufferedReSampler::processBlock(ScalarBuffer<double> const& input,
                                        ScalarBuffer<double>& output,
                                        int requiredSamples)
{
  output.setNumChannelsAndSamples(input.getNumChannels(), input.getNumSamples());
  processBlock(input, output.get(), output.getNumChannels(), requiredSamples);
}

int FirUnbufferedReSampler::processBlock(double* const* input,
                                         int numInputChannels,
                                         int numSamples,
                                         ScalarBuffer<double>& output)
{
  assert(numInputChannels <= numChannels);
  int numOutputSamples = (int)std::ceil(numSamples * oversamplingRate);
  if (output.getNumChannels() < numInputChannels || output.getCapacity() < numOutputSamples) {
    DEBUG_MESSAGE("A FirUnbufferedReSampler object had to allocate memory! Has "
                  "prepareBuffers been called?\n");
  }
  output.setNumChannelsAndSamples(numInputChannels, numOutputSamples);

  if (oversamplingRate == 1) {
    for (int c = 0; c < numInputChannels; ++c) {
      std::copy(&input[c][0], &input[c][0] + numSamples, &output[c][0]);
    }
    return numSamples;
  }
  int totalUpsampledSamples = 0;
  for (int c = 0; c < numInputChannels; ++c) {
    double* outPtr;
    int numInputSamples = numSamples;
    int inputCounter = 0;
    int outputCounter = 0;
    while (numInputSamples > 0) {
      int samplesToProcess = std::min(numInputSamples, maxSamplesPerBlock);
      int numUpsampledSamples = reSamplers[c]->process(&input[c][inputCounter], samplesToProcess, outPtr);
      inputCounter += samplesToProcess;
      numInputSamples -= samplesToProcess;
      if (numUpsampledSamples > 0) {
        if (outputCounter + numUpsampledSamples > numOutputSamples) {
          DEBUG_MESSAGE("A FirUnbufferedReSampler object had to allocate memory due to a "
                        "fluctuation, this shold not happen!\n");
          output.setNumSamples(outputCounter + numOutputSamples);
        }
        std::copy(outPtr, outPtr + numUpsampledSamples, &output[c][outputCounter]);
        outputCounter += numUpsampledSamples;
      }
    }
    totalUpsampledSamples = outputCounter;
  }
  return totalUpsampledSamples;
}
void FirBufferedReSampler::processBlock(double* const* input,
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
            DEBUG_MESSAGE("A FirBufferedReSampler object had to allocate "
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
        int numUpsampledSamples =
          reSamplers[c]->process(const_cast<double*>(&input[c][inputCounter]), samplesToProcess, outPtr);
        inputCounter += samplesToProcess;
        numInputSamples -= samplesToProcess;
        int neededBufferSize = bufferCounter + outputCounter + numUpsampledSamples;
        if (buffer.getNumSamples() < neededBufferSize) {
          if (buffer.getCapacity() < neededBufferSize) {
            DEBUG_MESSAGE("A FirBufferedReSampler object had to allocate "
                          "memory! Has prepareBuffers been called?");
          }
          buffer.setNumSamples(bufferCounter + outputCounter + numUpsampledSamples);
        }
        if (numUpsampledSamples > 0) {
          std::copy(outPtr, outPtr + numUpsampledSamples, &buffer[c][bufferCounter + outputCounter]);
          outputCounter += numUpsampledSamples;
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

void FirBufferedReSampler::processBlock(ScalarBuffer<double> const& input,
                                        double** output,
                                        int numOutputChannels,
                                        int requiredSamples)
{
  processBlock(input.get(), input.getNumSamples(), output, numOutputChannels, requiredSamples);
}

void FirReSamplerBase::setup()
{
  reSamplers.clear();

  if (numChannels == 0) {
    return;
  }

  for (int c = 0; c < numChannels; ++c) {
    reSamplers.push_back(
      std::make_unique<r8b::CDSPResampler24>(1.0, oversamplingRate, maxSamplesPerBlock, transitionBand));
  }
}

FirReSamplerBase::FirReSamplerBase(int numChannels,
                                   double transitionBand,
                                   int maxSamplesPerBlock,
                                   double oversamplingRate)
  : numChannels(numChannels)
  , maxSamplesPerBlock(maxSamplesPerBlock)
  , maxInputLength(maxSamplesPerBlock)
  , transitionBand(transitionBand)
  , oversamplingRate(oversamplingRate)
{}

FirBufferedReSampler::FirBufferedReSampler(int numChannels,
                                           double transitionBand,
                                           int maxSamplesPerBlock,
                                           double oversamplingRate)
  : FirReSamplerBase(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
  , maxRequiredOutputLength(maxSamplesPerBlock)
{
  setup();
}

FirUnbufferedReSampler::FirUnbufferedReSampler(int numChannels,
                                               double transitionBand,
                                               int maxSamplesPerBlock,
                                               double oversamplingRate)
  : FirReSamplerBase(numChannels, transitionBand, maxSamplesPerBlock, oversamplingRate)
{
  setup();
}

void FirReSamplerBase::setNumChannels(int value)
{
  numChannels = value;
  setup();
}

void FirReSamplerBase::setRate(double value)
{
  oversamplingRate = value;
  setup();
}

void FirReSamplerBase::setMaxSamplesPerBlock(int value)
{
  maxSamplesPerBlock = value;
  setup();
}

void FirReSamplerBase::setTransitionBand(int value)
{
  transitionBand = value;
  setup();
}

void FirReSamplerBase::reset()
{
  for (auto& reSampler : reSamplers) {
    reSampler->clear();
  }
}

void FirReSamplerBase::prepareBuffers(int numSamples)
{
  maxInputLength = numSamples;
  auto d = std::div(maxInputLength, maxSamplesPerBlock);
  int maxReSamplerOutputLength = reSamplers[0]->getMaxOutLen(maxSamplesPerBlock);
  maxOutputLength = (d.quot + (d.rem > 0 ? 1 : 0)) * maxReSamplerOutputLength;
}

void FirBufferedReSampler::prepareBuffers(int numInputSamples, int requiredOutputSamples)
{
  FirReSamplerBase::prepareBuffers(numInputSamples);
  maxRequiredOutputLength = requiredOutputSamples;
  int neededBufferSize = maxOutputLength + std::max(maxOutputLength, requiredOutputSamples);
  // maybe buffer.setNumSamples(maxOutputLength); is enough?
  if (buffer.getNumSamples() < neededBufferSize) {
    buffer.setNumSamples(neededBufferSize);
  }
}

void FirBufferedReSampler::reset()
{
  FirReSamplerBase::reset();
  bufferCounter = 0;
}

void FirUnbufferedReSampler::setup()
{
  FirReSamplerBase::setup();

  prepareBuffers(maxInputLength);
  reset();
}

void FirBufferedReSampler::setup()
{
  FirReSamplerBase::setup();

  buffer.setNumChannels(numChannels);
  prepareBuffers(maxInputLength, maxRequiredOutputLength);

  reset();
}

int FirReSamplerBase::getNumSamplesBeforeOutputStarts()
{
  if (reSamplers.size() == 0) {
    DEBUG_MESSAGE("Asking the number of samples before the output of the "
                  "reSamplers starts when there are 0 allocated reSamplers.");
    return 0;
  }
  return reSamplers[0]->getInLenBeforeOutStart();
}

} // namespace oversimple
