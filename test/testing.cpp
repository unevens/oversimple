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
#include "oversimple/IirOversampling.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

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

std::string n2s(double x)
{
  stringstream stream;
  stream << fixed << showpos << setprecision(6) << x;
  return stream.str();
}

std::string i2s(int x)
{
  stringstream stream;
  stream << setw(4) << setfill(' ') << x;
  return stream.str();
}

template<typename Scalar>
void testFirOversampling(int numChannels,
                         int numSamples,
                         int fftSamplesPerBlock,
                         int oversamplingOrder,
                         double transitionBand)
{
  cout << "testing Fir Oversampling with oversampling order " << oversamplingOrder << " and " << numChannels
       << " channels and " << numSamples << " samples per block"
       << " and " << fftSamplesPerBlock << " samples per fft block "
       << " and transitionBand = " << transitionBand << "%. with "
       << (std::is_same_v<Scalar, float> ? "single" : "double") << " precision"
       << "\n";
  auto firUpSampler =
    fir::TUpSamplerPreAllocated<Scalar>(oversamplingOrder, numChannels, transitionBand, fftSamplesPerBlock);
  auto firDownSampler =
    fir::TDownSamplerPreAllocated<Scalar>(oversamplingOrder, numChannels, transitionBand, fftSamplesPerBlock);
  firUpSampler.setOrder(oversamplingOrder);
  firDownSampler.setOrder(oversamplingOrder);
  firUpSampler.prepareBuffers(numSamples);
  auto const maxUpSampledSamples = firUpSampler.getMaxNumOutputSamples();
  firDownSampler.prepareBuffers(maxUpSampledSamples, numSamples);
  int upSampleLatency = firUpSampler.getNumSamplesBeforeOutputStarts();
  int downSampleLatency = firDownSampler.getNumSamplesBeforeOutputStarts();
  int latency = upSampleLatency + downSampleLatency / (1 << oversamplingOrder);
  cout << "NumSamplesBeforeUpSamplingStarts = " << upSampleLatency << "\n";
  cout << "NumSamplesBeforeDownSamplingStarts  = " << downSampleLatency << "\n";
  cout << "latency  = " << latency << "\n";
  auto const numBuffers = latency / numSamples + 2 * std::max(1, fftSamplesPerBlock / numSamples);
  auto const totSamples = numSamples * numBuffers;
  ScalarBuffer<Scalar> input(numChannels, totSamples);
  ScalarBuffer<Scalar> inputCopy(numChannels, totSamples);
  ScalarBuffer<Scalar> output(numChannels, totSamples);
  input.fill(0.0);
  inputCopy.fill(0.0);
  output.fill(0.0);

  for (int c = 0; c < numChannels; ++c) {
    for (int i = 0; i < inputCopy[c].size(); ++i) {
      inputCopy[c][i] = input[c][i] = sin(2.0 * M_PI * 0.125 * (Scalar)i);
    }
  }

  auto in = input.get();
  auto out = output.get();
  for (auto i = 0; i < numBuffers; ++i) {
    int numUpSampledSamples = firUpSampler.processBlock(in, numSamples);
    auto const& upSampled = firUpSampler.getOutput().get();
    CHECK_MEMORY;
    firDownSampler.processBlock(upSampled, numUpSampledSamples, out, numSamples);
    CHECK_MEMORY;
    //    cout << "numUpSampledSamples = " << numUpSampledSamples << "\n";
    for (auto c = 0; c < numChannels; ++c) {
      in[c] += numSamples;
      out[c] += numSamples;
    }
  }

  auto const measureSnr = [&](int from, int to, const char* text) {
    for (int c = 0; c < numChannels; ++c) {
      double noisePower = 0.0;
      double signalPower = 0.0;
      for (int i = from; i < to; ++i) {
        double in = inputCopy[c][i];
        double out = output[c][i + latency];
        //        cout << in << " | " << out << "\n";
        double diff = in - out;
        signalPower += in * in;
        noisePower += diff * diff;
      }

      cout << text << ": channel " << c << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
    }
  };

  measureSnr(0, fftSamplesPerBlock, "snr first block");
  measureSnr(fftSamplesPerBlock, totSamples - latency, "snr after first block");

  cout << "completed testing Fir Oversampling with oversampling order " << oversamplingOrder << " and " << numChannels
       << " channels and " << numSamples << " samples per block"
       << " and " << fftSamplesPerBlock << " samples per fft block "
       << " and transitionBand = " << transitionBand << "%. with "
       << (std::is_same_v<Scalar, float> ? "single" : "double") << " precision"
       << "\n";
}

template<typename Scalar>
void inspectIirOversampling(int numChannels, int samplesPerBlock, int order, int offset)
{
  int factor = (int)std::pow(2, order);
  cout << "beginning to test " << factor << "x "
       << "IirOversampling with " << numChannels << "channels and "
       << (typeid(Scalar) == typeid(float) ? "single" : "double") << " precision\n";
  // prepare data
  Scalar normalizedFrequency = 0.125;
  Scalar frequency = 2.0 * 3.141592653589793238 * normalizedFrequency;
  auto** in = new Scalar*[numChannels];
  auto upSampled = InterleavedBuffer<Scalar>(numChannels, factor * samplesPerBlock);
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
  auto upSampling = iir::UpSampler<Scalar>(numChannels, order);
  auto downSampling = iir::DownSampler<Scalar>(numChannels, order);
  bool const upSamplingOk = upSampling.setOrder(order);
  assert(upSamplingOk);
  bool const downSamplingOk = downSampling.setOrder(order);
  assert(downSamplingOk);
  upSampling.prepareBuffer(samplesPerBlock);
  downSampling.prepareBuffer(samplesPerBlock * (1 << order));
  CHECK_MEMORY;
  upSampling.processBlock(in, samplesPerBlock, upSampled);
  CHECK_MEMORY;
  downSampling.processBlock(upSampled, factor * samplesPerBlock);
  CHECK_MEMORY;
  auto& output = downSampling.getOutput();
  auto preset = iir::detail::getOversamplingPreset(order);
  double groupDelay = 2.0 * preset.getGroupDelay(normalizedFrequency, order);
  double phaseDelay = 2.0 * preset.getPhaseDelay(normalizedFrequency, order);
  for (int i = 0; i < numChannels; ++i) {
    cout << "channel " << i << "\n";
    for (int s = 0; s < samplesPerBlock; ++s) {
      cout << i2s(s) << ":   " << n2s(in[i][s]) << "  |  " << n2s(*output.at(i, s)) << "\n";
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
       << (typeid(Scalar) == typeid(float) ? "single" : "double") << " precision\n";
}

int main()
{
  if constexpr (AVEC_AVX512) {
    cout << "AVX512 AVAILABLE\n";
  }
  else if constexpr (AVEC_AVX) {
    cout << "AVX AVAILABLE\n";
  }
  else if constexpr (AVEC_SSE2) {
    cout << "SSE2 AVAILABLE\n";
  }
  else if constexpr (AVEC_NEON_64) {
    cout << "NEON WITH 64 BIT AVAILABLE\n";
  }
  else if constexpr (AVEC_NEON) {
    cout << "NEON WITH 32 BIT AVAILABLE\n";
  }
  else {
    cout << "NO SIMD INSTRUCTIONS AVAILABLE\n";
  }

  inspectIirOversampling<double>(2, 256, 4, 128);
  inspectIirOversampling<float>(2, 256, 4, 128);
  testFirOversampling<float>(2, 128, 1024, 4, 4.0);
  testFirOversampling<float>(2, 1024, 512, 4, 4.0);
  testFirOversampling<double>(2, 128, 1024, 4, 4.0);
  testFirOversampling<double>(2, 1024, 512, 4, 4.0);
  return 0;
}