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
#if (defined _DEBUG) && (defined _WIN32)
#include <Windows.h>
inline void
DEBUG_MESSAGE(char const* message)
{
  OutputDebugString(message);
}
#else
#define DEBUG_MESSAGE(x) /*nohing*/
#endif

namespace oversimple {

int
FirUnbufferedResampler::ProcessBlock(ScalarBuffer<double> const& input,
                                     ScalarBuffer<double>& output)
{
  return ProcessBlock(
    input.Get(), input.GetNumChannels(), input.GetSize(), output);
}

void
FirBufferedResampler::ProcessBlock(ScalarBuffer<double> const& input,
                                   ScalarBuffer<double>& output,
                                   int requiredSamples)
{
  output.SetNumChannelsAndSize(input.GetNumChannels(), input.GetSize());
  ProcessBlock(input, output.Get(), output.GetNumChannels(), requiredSamples);
}

int
FirUnbufferedResampler::ProcessBlock(double* const* input,
                                     int numInputChannels,
                                     int numSamples,
                                     ScalarBuffer<double>& output)
{
  assert(numInputChannels <= numChannels);
  int numOutputSamples = std::ceil(numSamples * oversamplingFactor);
  if (output.GetNumChannels() < numInputChannels ||
      output.GetCapacity() < numOutputSamples) {
    DEBUG_MESSAGE("A FirUnbufferedResampler object had to allocate memory! Has "
                  "PrepareBuffers been called?\n");
  }
  output.SetNumChannelsAndSize(numInputChannels, numOutputSamples);

  if (oversamplingFactor == 1) {
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
      int numUpsampledSamples = resamplers[c]->process(
        &input[c][inputCounter], samplesToProcess, outPtr);
      inputCounter += samplesToProcess;
      numInputSamples -= samplesToProcess;
      if (numUpsampledSamples > 0) {
        if (outputCounter + numUpsampledSamples > numOutputSamples) {
          DEBUG_MESSAGE(
            "A FirUnbufferedResampler object had to allocate memory due to a "
            "fluctuation, this shold not happen!\n");
          output.SetSize(outputCounter + numOutputSamples);
        }
        std::copy(
          outPtr, outPtr + numUpsampledSamples, &output[c][outputCounter]);
        outputCounter += numUpsampledSamples;
      }
    }
    totalUpsampledSamples = outputCounter;
  }
  return totalUpsampledSamples;
}

void
FirBufferedResampler::ProcessBlock(ScalarBuffer<double> const& input,
                                   double** output,
                                   int numOutputChannels,
                                   int requiredSamples)
{
  assert(numOutputChannels <= numChannels);
  int numSamples = input.GetSize();

  if (oversamplingFactor == 1.0) {
    for (int c = 0; c < numOutputChannels; ++c) {
      std::copy(&input[c][0], &input[c][0] + numSamples, &output[c][0]);
    }
    return;
  }

  int newBufferCounter = bufferCounter;
  if (numSamples <= maxSamplesPerBlock) {
    for (int c = 0; c < numOutputChannels; ++c) {
      double* outPtr;
      int numUpsampledSamples = resamplers[c]->process(
        const_cast<double*>(&input[c][0]), numSamples, outPtr);
      int diff = requiredSamples - numUpsampledSamples - bufferCounter;
      if (diff >= 0) {
        std::fill_n(&output[c][0], diff, 0.0);
        std::copy(
          &buffer[c][0], &buffer[c][0] + bufferCounter, &output[c][diff]);
        std::copy(outPtr,
                  outPtr + numUpsampledSamples,
                  &output[c][diff + bufferCounter]);
        newBufferCounter = 0;
      }
      else { // diff < 0
        int samplesFromBuffer = std::min(bufferCounter, requiredSamples);
        std::copy(
          &buffer[c][0], &buffer[c][0] + samplesFromBuffer, &output[c][0]);
        std::copy(&buffer[c][0] + samplesFromBuffer,
                  &buffer[c][0] + bufferCounter,
                  &buffer[c][0]);
        newBufferCounter = bufferCounter - samplesFromBuffer;
        int wantedSamplesFromResampler = requiredSamples - samplesFromBuffer;
        int samplesFromResampler =
          std::min(wantedSamplesFromResampler, numUpsampledSamples);
        std::copy(
          outPtr, outPtr + samplesFromResampler, &output[c][samplesFromBuffer]);
        // check buffer size
        int neededBufferSize =
          newBufferCounter + numUpsampledSamples - samplesFromResampler;
        if (buffer.GetSize() < neededBufferSize) {
          if (buffer.GetCapacity() < neededBufferSize) {
            DEBUG_MESSAGE("A FirBufferedResampler object had to allocate "
                          "memory! Has PrepareBuffers been called?\n");
          }
          buffer.SetSize(neededBufferSize);
        }
        // copy tail to buffer
        std::copy(outPtr + samplesFromResampler,
                  outPtr + numUpsampledSamples,
                  &buffer[c][newBufferCounter]);
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
          resamplers[c]->process(const_cast<double*>(&input[c][inputCounter]),
                                 samplesToProcess,
                                 outPtr);
        inputCounter += samplesToProcess;
        numInputSamples -= samplesToProcess;
        int neededBufferSize =
          bufferCounter + outputCounter + numUpsampledSamples;
        if (buffer.GetSize() < neededBufferSize) {
          if (buffer.GetCapacity() < neededBufferSize) {
            DEBUG_MESSAGE("A FirBufferedResampler object had to allocate "
                          "memory! Has PrepareBuffers been called?");
          }
          buffer.SetSize(bufferCounter + outputCounter + numUpsampledSamples);
        }
        if (numUpsampledSamples > 0) {
          std::copy(outPtr,
                    outPtr + numUpsampledSamples,
                    &buffer[c][bufferCounter + outputCounter]);
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
        std::copy(
          &buffer[c][0], &buffer[c][0] + bufferCounter, &output[c][diff]);
      }
      bufferCounter = 0;
    }
    else { // diff < 0
      for (int c = 0; c < numOutputChannels; ++c) {
        std::copy(
          &buffer[c][0], &buffer[c][0] + requiredSamples, &output[c][0]);
        std::copy(&buffer[c][0] + requiredSamples,
                  &buffer[c][0] + bufferCounter,
                  &buffer[c][0]);
      }
    }
  }
}

void
FirResamplerBase::Setup()
{
  resamplers.clear();

  if (numChannels == 0) {
    return;
  }

  for (int c = 0; c < numChannels; ++c) {
    resamplers.push_back(std::make_unique<r8b::CDSPResampler24>(
      1.0, oversamplingFactor, maxSamplesPerBlock, transitionBand));
  }
}

FirResamplerBase::FirResamplerBase(int numChannels,
                                   double transitionBand,
                                   int maxSamplesPerBlock,
                                   double oversamplingFactor)
  : numChannels(numChannels)
  , maxSamplesPerBlock(maxSamplesPerBlock)
  , maxInputLength(maxSamplesPerBlock)
  , transitionBand(transitionBand)
  , oversamplingFactor(oversamplingFactor)
{}

FirBufferedResampler::FirBufferedResampler(int numChannels,
                                           double transitionBand,
                                           int maxSamplesPerBlock,
                                           double oversamplingFactor)
  : FirResamplerBase(numChannels,
                     transitionBand,
                     maxSamplesPerBlock,
                     oversamplingFactor)
  , maxRequiredOutputLength(maxSamplesPerBlock)
{
  Setup();
}

FirUnbufferedResampler::FirUnbufferedResampler(int numChannels,
                                               double transitionBand,
                                               int maxSamplesPerBlock,
                                               double oversamplingFactor)
  : FirResamplerBase(numChannels,
                     transitionBand,
                     maxSamplesPerBlock,
                     oversamplingFactor)
{
  Setup();
}

void
FirResamplerBase::SetNumChannels(int value)
{
  numChannels = value;
  Setup();
}

void
FirResamplerBase::SetOversamplingFactor(double value)
{
  oversamplingFactor = value;
  Setup();
}

void
FirResamplerBase::SetMaxSamplesPerBlock(int value)
{
  maxSamplesPerBlock = value;
  Setup();
}

void
FirResamplerBase::SetTransitionBand(int value)
{
  transitionBand = value;
  Setup();
}

void
FirResamplerBase::Reset()
{
  for (auto& resampler : resamplers) {
    resampler->clear();
  }
}

void
FirResamplerBase::PrepareBuffers(int numSamples)
{
  maxInputLength = numSamples;
  auto d = std::div(maxInputLength, maxSamplesPerBlock);
  int maxResamplerOutputLength =
    resamplers[0]->getMaxOutLen(maxSamplesPerBlock);
  maxOutputLength = (d.quot + (d.rem > 0 ? 1 : 0)) * maxResamplerOutputLength;
}

void
FirBufferedResampler::PrepareBuffers(int numInputSamples,
                                     int requiredOutputSamples)
{
  FirResamplerBase::PrepareBuffers(numInputSamples);
  maxRequiredOutputLength = requiredOutputSamples;
  int neededBufferSize =
    maxOutputLength + std::max(maxOutputLength, requiredOutputSamples);
  // maybe buffer.SetSize(maxOutputLength); is enough?
  if (buffer.GetSize() < neededBufferSize) {
    buffer.SetSize(neededBufferSize);
  }
}

void
FirBufferedResampler::Reset()
{
  FirResamplerBase::Reset();
  bufferCounter = 0;
}

void
FirUnbufferedResampler::Setup()
{
  FirResamplerBase::Setup();

  PrepareBuffers(maxInputLength);
  Reset();
}

void
FirBufferedResampler::Setup()
{
  FirResamplerBase::Setup();

  buffer.SetNumChannels(numChannels);
  PrepareBuffers(maxInputLength, maxRequiredOutputLength);

  Reset();
}

int
FirResamplerBase::GetNumSamplesBeforeOutputStarts()
{
  if (resamplers.size() == 0) {
    DEBUG_MESSAGE("Asking the number of samples before the output of the "
                  "resamplers starts when there are 0 allocated resamplers.");
    return 0;
  }
  return resamplers[0]->getInLenBeforeOutStart();
}

} // namespace oversimple
