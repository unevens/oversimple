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
#include "oversimple/IiroversamplingFactory.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// macro-paranoia macro
#ifdef _MSC_VER
#ifdef _DEBUG
#ifndef CHECK_MEMORY
#define _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC_NEW
#include <crtdbg.h>
#define CHECK_MEMORY assert(_CrtCheckMemory());
#endif
#else
#define CHECK_MEMORY /*nothing*/
#endif
#else
#define CHECK_MEMORY /*nothing*/
#endif

using namespace oversimple;
using namespace std;

std::string
n2s(double x)
{
  stringstream stream;
  stream << fixed << showpos << setprecision(6) << x;
  return stream.str();
}

std::string
i2s(int x)
{
  stringstream stream;
  stream << setw(4) << setfill(' ') << x;
  return stream.str();
}

template<typename Scalar>
void
testFirOversampler(int numChannels,
                   int samplesPerBlock,
                   int oversamplingRate,
                   double transitionBand)
{
  cout << "testing FirOversampler with oversamplingRate "
       << oversamplingRate << " and " << numChannels << " channels and "
       << samplesPerBlock << " samples per block"
       << " and transitionBand = " << transitionBand << "%"
       << "\n";
  TFirUpsampler<Scalar> firUpsampler(numChannels, transitionBand);
  TFirDownsampler<Scalar> firDownsampler(numChannels, transitionBand);
  firDownsampler.setRate(oversamplingRate);
  firUpsampler.setRate(oversamplingRate);
  int upsamplePadding = firUpsampler.getNumSamplesBeforeOutputStarts();
  int downsamplePadding = firDownsampler.getNumSamplesBeforeOutputStarts();
  int padding = upsamplePadding + downsamplePadding;
  cout << "NumSamplesBeforeUpsamplingStarts = " << upsamplePadding << "\n";
  cout << "NumSamplesBeforeDownsamplingStarts  = " << downsamplePadding << "\n";

  ScalarBuffer<Scalar> input(numChannels, samplesPerBlock + padding);
  ScalarBuffer<Scalar> inputCopy(numChannels, samplesPerBlock + padding);
  ScalarBuffer<Scalar> output(numChannels, samplesPerBlock + padding);
  ScalarBuffer<Scalar> upsampled(
    numChannels, samplesPerBlock * oversamplingRate + padding);
  input.fill(0.0);
  inputCopy.fill(0.0);
  output.fill(0.0);
  upsampled.fill(0.0);

  for (int c = 0; c < numChannels; ++c) {
    for (int i = 0; i < samplesPerBlock; ++i) {
      inputCopy[c][i] = input[c][i] = sin(2.0 * M_PI * 0.125 * (Scalar)i);
    }
  }

  int numUpsampledSamples = firUpsampler.processBlock(input, upsampled);
  CHECK_MEMORY;
  firDownsampler.processBlock(upsampled, output, samplesPerBlock);
  CHECK_MEMORY;
  cout << "numUpsampledSamples = " << numUpsampledSamples << "\n";

  for (int c = 0; c < numChannels; ++c) {
    double noisePower = 0.0;
    double signalPower = 0.0;
    for (int i = samplesPerBlock / 2; i < samplesPerBlock; ++i) {
      double in = inputCopy[c][i];
      double out = output[c][i];
      // cout << in << " | " << out << "\n";
      double diff = in - out;
      signalPower += in * in;
      noisePower += diff * diff;
    }
    signalPower /= 0.5 * (double)samplesPerBlock;
    noisePower /= 0.5 * (double)samplesPerBlock;

    cout << "channel " << c
         << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
  }
  CHECK_MEMORY;

  cout << "completed testing FirOversampler with oversamplingRate "
       << oversamplingRate << " and " << numChannels << " channels and "
       << samplesPerBlock << " samples per block"
       << " and transitionBand = " << transitionBand << "%"
       << "\n";
}

template<typename Scalar>
void
inspectIirOversampling(int numChannels,
                       int samplesPerBlock,
                       int order,
                       int offset)
{
  int factor = (int)std::pow(2, order);
  cout << "beginning to test " << factor << "x "
       << "IirOversampling with " << numChannels << "channels and "
       << (typeid(Scalar) == typeid(float) ? "single" : "double")
       << " precision\n";
  // prepare data
  // Scalar normalizedFrequency =  0.05;
  Scalar normalizedFrequency = 0.125;
  Scalar frequency = 2.0 * 3.141592653589793238 * normalizedFrequency;
  Scalar** in = new Scalar*[numChannels];
  for (int i = 0; i < numChannels; ++i) {
    in[i] = new Scalar[samplesPerBlock];
    for (int s = 0; s < offset; ++s) {
      in[i][s] = 0.0;
    }
    if (i % 2 == 0) {
      for (int s = offset; s < samplesPerBlock; ++s) {
        in[i][s] = std::sin(frequency * (Scalar)(s - offset));
      }
    }
    else {
      for (int s = offset; s < 2 * offset; ++s) {
        in[i][s] = 1.0;
      }
      for (int s = 2 * offset; s < samplesPerBlock; ++s) {
        in[i][s] = 0.0;
      }
    }
  }
  // Oversampling test
  auto upsampling = IirUpsamplerFactory<Scalar>::make(numChannels);
  auto downsampling = IirDownsamplerFactory<Scalar>::make(numChannels);
  upsampling->setOrder(order);
  downsampling->setOrder(order);
  auto buffer =
    InterleavedBuffer<Scalar>(numChannels, factor * samplesPerBlock);
  CHECK_MEMORY;
  upsampling->processBlock(in, samplesPerBlock, buffer);
  CHECK_MEMORY;
  downsampling->processBlock(buffer, factor * samplesPerBlock);
  CHECK_MEMORY;
  auto& output = downsampling->getOutput();
  auto preset = getIirOversamplingPreset(order);
  double groupDelay = 2.0 * preset.getGroupDelay(normalizedFrequency, order);
  double phaseDelay = 2.0 * preset.getPhaseDelay(normalizedFrequency, order);
  for (int i = 0; i < numChannels; ++i) {
    cout << "channel " << i << "\n";
    for (int s = 0; s < samplesPerBlock; ++s) {
      cout << i2s(s) << ":   " << n2s(in[i][s]) << "  |  "
           << n2s(*output.at(i, s)) << "\n";
    }
    cout << "\n";
  }
  cout << "IirOversampling test completed\n";
  CHECK_MEMORY;
  // cleanup
  for (int i = 0; i < numChannels; ++i) {
    delete[] in[i];
  }
  delete[] in;
  cout << "completed testing " << factor << "x "
       << "IirOversampling with " << numChannels << "channels and "
       << (typeid(Scalar) == typeid(float) ? "single" : "double")
       << " precision\n";
}

int
main()
{
  cout << "avx? " << AVX_AVAILABLE << "\n";

  inspectIirOversampling<double>(2, 1024, 4, 128);
  inspectIirOversampling<float>(2, 1024, 4, 128);
  testFirOversampler<double>(2, 16384, 16, 4.0);
  testFirOversampler<float>(2, 16384, 16, 4.0);
  return 0;
}