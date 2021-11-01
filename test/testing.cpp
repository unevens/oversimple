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
#include "oversimple/Oversampling.hpp"

#include <cmath>
#include <iostream>
#include <optional>

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

template<typename Float>
void testFirOversampling(uint64_t numChannels,
                         uint64_t numSamples,
                         uint64_t fftSamplesPerBlock,
                         uint64_t oversamplingOrder,
                         double transitionBand)
{
  cout << "\n";
  cout << "\n";
  cout << "testing Fir Oversampling with oversampling order " << oversamplingOrder << " and " << numChannels
       << " channels and " << numSamples << " samples per block"
       << " and " << fftSamplesPerBlock << " samples per fft block "
       << " and transitionBand = " << transitionBand << "%. with "
       << (std::is_same_v<Float, float> ? "single" : "double") << " precision"
       << "\n";
  auto firUpSampler = fir::TUpSamplerPreAllocated<Float>(oversamplingOrder, 1, transitionBand, fftSamplesPerBlock);
  auto firDownSampler = fir::TDownSamplerPreAllocated<Float>(oversamplingOrder, 1, transitionBand, fftSamplesPerBlock);
  firUpSampler.setNumChannels(numChannels);
  firUpSampler.setOrder(oversamplingOrder);
  firUpSampler.prepareBuffers(numSamples);
  auto const maxUpSampledSamples = firUpSampler.getMaxNumOutputSamples();
  firDownSampler.setNumChannels(numChannels);
  firDownSampler.setOrder(oversamplingOrder);
  firDownSampler.prepareBuffers(maxUpSampledSamples, numSamples);
  uint64_t upSampleLatency = firUpSampler.getNumSamplesBeforeOutputStarts();
  uint64_t downSampleLatency = firDownSampler.getNumSamplesBeforeOutputStarts();
  uint64_t latency = upSampleLatency + downSampleLatency / (1 << oversamplingOrder);
  cout << "NumSamplesBeforeUpSamplingStarts = " << upSampleLatency << "\n";
  cout << "NumSamplesBeforeDownSamplingStarts  = " << downSampleLatency << "\n";
  cout << "latency  = " << latency << "\n";
  auto const numBuffers = latency / numSamples + 2 * std::max(fftSamplesPerBlock / numSamples, (uint64_t)1);
  auto const totSamples = numSamples * numBuffers;
  Buffer<Float> input(numChannels, totSamples);
  Buffer<Float> output(numChannels, totSamples);
  input.fill(0.0);
  output.fill(0.0);

  for (uint64_t c = 0; c < numChannels; ++c) {
    for (uint64_t i = 0; i < input[c].size(); ++i) {
      input[c][i] = sin(2.0 * M_PI * 0.125 * (Float)i);
    }
  }

  auto in = input.get();
  auto out = output.get();
  for (auto i = 0; i < numBuffers; ++i) {
    uint64_t numUpSampledSamples = firUpSampler.processBlock(in, numSamples);
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

  auto const measureSnr = [&](uint64_t from, uint64_t to, const char* text) {
    for (uint64_t c = 0; c < numChannels; ++c) {
      double noisePower = 0.0;
      double signalPower = 0.0;
      for (uint64_t i = from; i < to; ++i) {
        double in = input[c][i];
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
       << (std::is_same_v<Float, float> ? "single" : "double") << " precision"
       << "\n";
}

template<typename Float>
void testIirOversampling(uint64_t numChannels, uint64_t order, uint64_t numSamples)
{
  cout << "\n";
  cout << "\n";
  auto const preset = iir::detail::getOversamplingPreset(0);
  double const groupDelay = 2 * preset.getGroupDelay(0, order);

  auto const factor = (uint64_t)std::pow(2, order);
  cout << "beginning to test " << factor << "x "
       << "IirOversampling with " << numChannels << "channels and "
       << (typeid(Float) == typeid(float) ? "single" : "double") << " precision\n";

  cout << "group delay at DC is " << groupDelay << "\n";
  auto const offset = 20 * (uint32_t)std::ceil(groupDelay);
  auto const samplesPerBlock = offset + numSamples;

  auto in = Buffer<Float>(numChannels, samplesPerBlock);
  in.fill(1.0);
  // Oversampling test
  auto upSampling = iir::UpSampler<Float>(1, order);
  upSampling.setNumChannels(numChannels);
  auto downSampling = iir::DownSampler<Float>(1, order);
  downSampling.setNumChannels(numChannels);
  bool const upSamplingOk = upSampling.setOrder(order);
  assert(upSamplingOk);
  bool const downSamplingOk = downSampling.setOrder(order);
  assert(downSamplingOk);
  upSampling.prepareBuffers(samplesPerBlock);
  downSampling.prepareBuffers(samplesPerBlock);
  CHECK_MEMORY;
  upSampling.processBlock(in);
  CHECK_MEMORY;
  auto const& upSampled = upSampling.getOutput();
  downSampling.processBlock(upSampled);
  CHECK_MEMORY;
  auto& output = downSampling.getOutput();

  auto const measureSnr = [&](uint64_t offset, uint64_t from, uint64_t to, const char* text) {
    for (uint64_t c = 0; c < numChannels; ++c) {
      double noisePower = 0.0;
      double signalPower = 0.0;
      for (uint64_t s = from; s < to; ++s) {
        double out = *output.at(c, s + offset);
        //        cout << s << " | " << in[c][s] << " | " << out << "\n";
        double diff = in[c][s] - out;
        signalPower += in[c][s] * in[c][s];
        noisePower += diff * diff;
      }
      cout << text << ": channel " << c << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
    }
  };
  measureSnr(groupDelay, 0, offset, "IIR snr up to 20x group delay");
  measureSnr(offset, 0, numSamples - offset, "IIR snr after 20x group delay");

  cout << "IirOversampling test completed\n";
  CHECK_MEMORY;

  cout << "completed testing " << factor << "x "
       << "IirOversampling with " << numChannels << "channels and "
       << (typeid(Float) == typeid(float) ? "single" : "double") << " precision\n";
}

template<typename Float>
void testOversampling(uint64_t order, uint64_t numSamples, bool linearPhase)
{

  OversamplingSettings settings;
  settings.maxOrder = order;
  settings.order = order;
  settings.numUpSampledChannels = 2;
  settings.numDownSampledChannels = 2;
  cout << "\n";
  cout << "\n";
  cout << "Testing wrapper with order" << order << " and numSamples " << numSamples << "\n";

  cout << "linear phase"
       << "\n";
  settings.isUsingLinearPhase = linearPhase;
  {
    cout << "\n";
    settings.upSampleInputBufferType = BufferType::plain;
    settings.downSampleInputBufferType = settings.upSampleOutputBufferType = BufferType::plain;
    settings.downSampleOutputBufferType = BufferType::plain;
    cout << "up-sampler input type = "
         << "plain"
         << "\n";
    cout << "up-sampler output and down-sampler input type = "
         << "plain"
         << "\n";
    cout << "down-sampler output type = "
         << "plain"
         << "\n";
    auto oversampling = Oversampling<Float>{ settings };
    oversampling.prepareBuffers(numSamples);
    auto const latency = oversampling.getLatency();
    cout << "latency = " << latency << "\n";
    cout << "up-sampling latency = " << oversampling.getUpSamplingLatency() << "\n";
    cout << "down-sampling latency = " << oversampling.getDownSamplingLatency() << "\n";
    auto const numBuffers = latency / numSamples + 2 * std::max(settings.fftBlockSize / numSamples, (uint64_t)1);
    auto const totSamples = numSamples * numBuffers;
    Buffer<Float> input(settings.numUpSampledChannels, totSamples);
    Buffer<Float> output(settings.numDownSampledChannels, totSamples);
    input.fill(0.0);
    output.fill(0.0);

    for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c) {
      for (uint64_t i = 0; i < input[c].size(); ++i) {
        input[c][i] = linearPhase ? sin(2.0 * M_PI * 0.125 * (Float)i) : 1.0;
      }
    }

    auto in = input.get();
    auto out = output.get();
    for (auto i = 0; i < numBuffers; ++i) {
      auto const numUpSampledSamples = oversampling.upSample(in, numSamples);
      auto const& upSampled = oversampling.getUpSampleOutput();
      assert(upSampled.getNumChannels() == settings.numUpSampledChannels);
      assert(upSampled.getNumSamples() == numUpSampledSamples);
      CHECK_MEMORY;
      oversampling.downSample(upSampled.get(), upSampled.getNumSamples(), out, numSamples);
      CHECK_MEMORY;
      // cout << "numUpSampledSamples = " << numUpSampledSamples << "\n";
      for (auto c = 0; c < settings.numUpSampledChannels; ++c) {
        in[c] += numSamples;
      }
      for (auto c = 0; c < settings.numDownSampledChannels; ++c) {
        out[c] += numSamples;
      }
    }

    auto const measureSnr = [&](int from, int to, const char* text) {
      for (int c = 0; c < settings.numDownSampledChannels; ++c) {
        double noisePower = 0.0;
        double signalPower = 0.0;
        for (int i = from; i < to; ++i) {
          double in = input[c][i];
          double out = output[c][i + latency];
          //            cout << in << " | " << out << "\n";
          double diff = in - out;
          signalPower += in * in;
          noisePower += diff * diff;
        }

        cout << text << ": channel " << c << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
      }
    };

    measureSnr(0, settings.fftBlockSize, "snr first block");
    measureSnr(settings.fftBlockSize, totSamples - latency, "snr after first block");
  }

  {
    cout << "\n";
    settings.upSampleInputBufferType = BufferType::plain;
    settings.downSampleInputBufferType = settings.upSampleOutputBufferType = BufferType::interleaved;
    settings.downSampleOutputBufferType = BufferType::plain;
    cout << "up-sampler input type = "
         << "plain"
         << "\n";
    cout << "up-sampler output and down-sampler input type = "
         << "interleaved"
         << "\n";
    cout << "down-sampler output type = "
         << "plain"
         << "\n";
    auto oversampling = Oversampling<Float>{ settings };
    oversampling.prepareBuffers(numSamples);
    auto const latency = oversampling.getLatency();
    cout << "latency = " << latency << "\n";
    cout << "up-sampling latency = " << oversampling.getUpSamplingLatency() << "\n";
    cout << "down-sampling latency = " << oversampling.getDownSamplingLatency() << "\n";
    auto const numBuffers = latency / numSamples + 2 * std::max(settings.fftBlockSize / numSamples, (uint64_t)1);
    auto const totSamples = numSamples * numBuffers;
    Buffer<Float> input(settings.numUpSampledChannels, totSamples);
    Buffer<Float> output(settings.numDownSampledChannels, totSamples);
    input.fill(0.0);
    output.fill(0.0);

    for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c) {
      for (uint64_t i = 0; i < input[c].size(); ++i) {
        input[c][i] = linearPhase ? sin(2.0 * M_PI * 0.125 * (Float)i) : 1.0;
      }
    }

    auto in = input.get();
    auto out = output.get();
    for (auto i = 0; i < numBuffers; ++i) {
      auto const numUpSampledSamples = oversampling.upSample(in, numSamples);
      auto const& upSampled = oversampling.getUpSampleOutputInterleaved();
      assert(upSampled.getNumChannels() == settings.numUpSampledChannels);
      assert(upSampled.getNumSamples() == numUpSampledSamples);
      CHECK_MEMORY;
      oversampling.downSample(upSampled, out, numSamples);
      CHECK_MEMORY;
      // cout << "numUpSampledSamples = " << numUpSampledSamples << "\n";
      for (auto c = 0; c < settings.numUpSampledChannels; ++c) {
        in[c] += numSamples;
      }
      for (auto c = 0; c < settings.numDownSampledChannels; ++c) {
        out[c] += numSamples;
      }
    }

    auto const measureSnr = [&](int from, int to, const char* text) {
      for (int c = 0; c < settings.numDownSampledChannels; ++c) {
        double noisePower = 0.0;
        double signalPower = 0.0;
        for (int i = from; i < to; ++i) {
          double in = input[c][i];
          double out = output[c][i + latency];
          //            cout << in << " | " << out << "\n";
          double diff = in - out;
          signalPower += in * in;
          noisePower += diff * diff;
        }

        cout << text << ": channel " << c << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
      }
    };

    measureSnr(0, settings.fftBlockSize, "snr first block");
    measureSnr(settings.fftBlockSize, totSamples - latency, "snr after first block");
  }

  {
    cout << "\n";
    settings.upSampleInputBufferType = BufferType::interleaved;
    settings.downSampleInputBufferType = settings.upSampleOutputBufferType = BufferType::interleaved;
    settings.downSampleOutputBufferType = BufferType::interleaved;
    cout << "up-sampler input type = "
         << "interleaved"
         << "\n";
    cout << "up-sampler output and down-sampler input type = "
         << "interleaved"
         << "\n";
    cout << "down-sampler output type = "
         << "interleaved"
         << "\n";
    auto oversampling = Oversampling<Float>{ settings };
    oversampling.prepareBuffers(numSamples);
    auto const latency = oversampling.getLatency();
    cout << "latency = " << latency << "\n";
    cout << "up-sampling latency = " << oversampling.getUpSamplingLatency() << "\n";
    cout << "down-sampling latency = " << oversampling.getDownSamplingLatency() << "\n";
    auto const numBuffers = latency / numSamples + 2 * std::max(settings.fftBlockSize / numSamples, (uint64_t)1);
    auto const totSamples = numSamples * numBuffers;
    Buffer<Float> input(settings.numUpSampledChannels, totSamples);

    for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c) {
      for (uint64_t i = 0; i < input[c].size(); ++i) {
        input[c][i] = linearPhase ? sin(2.0 * M_PI * 0.125 * (Float)i) : 1.0;
      }
    }
    InterleavedBuffer<Float> inputInterleaved(settings.numUpSampledChannels, numSamples);
    auto inPtr = input.get();
    Buffer<Float> outputPlain(settings.numDownSampledChannels, totSamples);

    auto out = outputPlain.get();
    for (auto i = 0; i < numBuffers; ++i) {
      inputInterleaved.interleave(inPtr, settings.numUpSampledChannels, numSamples);
      auto const numUpSampledSamples = oversampling.upSample(inputInterleaved);
      auto const& upSampled = oversampling.getUpSampleOutputInterleaved();
      assert(upSampled.getNumChannels() == settings.numUpSampledChannels);
      assert(upSampled.getNumSamples() == numUpSampledSamples);
      CHECK_MEMORY;
      oversampling.downSample(upSampled, numSamples);
      CHECK_MEMORY;
      // cout << "numUpSampledSamples = " << numUpSampledSamples << "\n";
      oversampling.getDownSampleOutputInterleaved().deinterleave(out, settings.numDownSampledChannels, numSamples);
      for (auto c = 0; c < settings.numDownSampledChannels; ++c) {
        out[c] += numSamples;
      }
      for (auto c = 0; c < settings.numUpSampledChannels; ++c) {
        inPtr[c] += numSamples;
      }
    }
    auto const measureSnr = [&](int from, int to, const char* text) {
      for (int c = 0; c < settings.numDownSampledChannels; ++c) {
        double noisePower = 0.0;
        double signalPower = 0.0;
        for (int i = from; i < to; ++i) {
          double in = input[c][i];
          double out = outputPlain[c][i + latency];
          //          cout << in << " | " << out << "\n";
          double diff = in - out;
          signalPower += in * in;
          noisePower += diff * diff;
        }

        cout << text << ": channel " << c << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
      }
    };

    measureSnr(0, settings.fftBlockSize, "snr first block");
    measureSnr(settings.fftBlockSize, totSamples - latency, "snr after first block");
  }

  {
    cout << "\n";
    settings.upSampleInputBufferType = BufferType::interleaved;
    settings.downSampleInputBufferType = settings.upSampleOutputBufferType = BufferType::plain;
    settings.downSampleOutputBufferType = BufferType::interleaved;
    cout << "up-sampler input type = "
         << "interleaved"
         << "\n";
    cout << "up-sampler output and down-sampler input type = "
         << "interleaved"
         << "\n";
    cout << "down-sampler output type = "
         << "interleaved"
         << "\n";
    auto oversampling = Oversampling<Float>{ settings };
    oversampling.prepareBuffers(numSamples);
    auto const latency = oversampling.getLatency();
    cout << "latency = " << latency << "\n";
    cout << "up-sampling latency = " << oversampling.getUpSamplingLatency() << "\n";
    cout << "down-sampling latency = " << oversampling.getDownSamplingLatency() << "\n";
    auto const numBuffers = latency / numSamples + 2 * std::max(settings.fftBlockSize / numSamples, (uint64_t)1);
    auto const totSamples = numSamples * numBuffers;
    Buffer<Float> input(settings.numUpSampledChannels, totSamples);

    for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c) {
      for (uint64_t i = 0; i < input[c].size(); ++i) {
        input[c][i] = linearPhase ? sin(2.0 * M_PI * 0.125 * (Float)i) : 1.0;
      }
    }
    InterleavedBuffer<Float> inputInterleaved(settings.numUpSampledChannels, numSamples);
    auto inPtr = input.get();
    Buffer<Float> outputPlain(settings.numDownSampledChannels, totSamples);

    auto out = outputPlain.get();
    for (auto i = 0; i < numBuffers; ++i) {
      inputInterleaved.interleave(inPtr, settings.numUpSampledChannels, numSamples);
      auto const numUpSampledSamples = oversampling.upSample(inputInterleaved);
      auto const& upSampled = oversampling.getUpSampleOutput();
      assert(upSampled.getNumChannels() == settings.numUpSampledChannels);
      assert(upSampled.getNumSamples() == numUpSampledSamples);
      CHECK_MEMORY;
      oversampling.downSample(upSampled.get(), numUpSampledSamples, numSamples);
      CHECK_MEMORY;
      // cout << "numUpSampledSamples = " << numUpSampledSamples << "\n";
      oversampling.getDownSampleOutputInterleaved().deinterleave(out, settings.numDownSampledChannels, numSamples);
      for (auto c = 0; c < settings.numDownSampledChannels; ++c) {
        out[c] += numSamples;
      }
      for (auto c = 0; c < settings.numUpSampledChannels; ++c) {
        inPtr[c] += numSamples;
      }
    }
    auto const measureSnr = [&](int from, int to, const char* text) {
      for (int c = 0; c < settings.numDownSampledChannels; ++c) {
        double noisePower = 0.0;
        double signalPower = 0.0;
        for (int i = from; i < to; ++i) {
          double in = input[c][i];
          double out = outputPlain[c][i + latency];
          //          cout << in << " | " << out << "\n";
          double diff = in - out;
          signalPower += in * in;
          noisePower += diff * diff;
        }

        cout << text << ": channel " << c << " snr = " << 10.0 * log10(signalPower / noisePower) << " dB\n";
      }
    };

    measureSnr(0, settings.fftBlockSize, "snr first block");
    measureSnr(settings.fftBlockSize, totSamples - latency, "snr after first block");
  }
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

  testIirOversampling<double>(2, 4, 1024);
  testIirOversampling<float>(2, 4, 1024);
  testFirOversampling<float>(2, 128, 1024, 4, 4.0);
  testFirOversampling<float>(2, 1024, 512, 4, 4.0);
  testFirOversampling<double>(2, 128, 1024, 4, 4.0);
  testFirOversampling<double>(2, 1024, 512, 4, 4.0);

  testOversampling<float>(4, 1024, false);
  testOversampling<float>(4, 1024, true);
  testOversampling<double>(4, 1024, false);
  testOversampling<double>(4, 1024, true);
  return 0;
}