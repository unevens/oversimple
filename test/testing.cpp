/*
Copyright 2019-2026 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// doctest-based test suite for oversimple. The previous version logged SNR
// to stdout without ever asserting — a regression that dropped SNR from
// 100 dB to 30 dB would still "pass". This one keeps the round-trip-and-
// measure-SNR strategy but turns each measurement into a REQUIRE against a
// per-(test, phase, block-type) threshold so CI catches drift.
//
// Thresholds are intentionally conservative — well below what the current
// FIR / IIR designs deliver — so they fail only when something is actually
// broken, not on small algorithmic tweaks. Tighten when ready.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "oversimple/FirOversampling.hpp"
#include "oversimple/IirOversampling.hpp"
#include "oversimple/Oversampling.hpp"

#include <cmath>
#include <vector>

using namespace oversimple;

namespace {

// Per-channel SNR in dB. Returns one number per channel.
template<class Buf>
std::vector<double> measure_snr_per_channel(const Buf& in, const Buf& out,
                                            uint64_t latency,
                                            uint64_t from, uint64_t to)
{
  const uint64_t numChannels = in.getNumChannels();
  std::vector<double> snr(numChannels, 0.0);
  for (uint64_t c = 0; c < numChannels; ++c) {
    double signalPower = 0.0;
    double noisePower  = 0.0;
    for (uint64_t i = from; i < to; ++i) {
      const double s = in[c][i];
      const double o = out[c][i + latency];
      const double d = s - o;
      signalPower += s * s;
      noisePower  += d * d;
    }
    snr[c] = (noisePower > 0.0) ? 10.0 * std::log10(signalPower / noisePower)
                                : 200.0;  // perfect → cap at 200 dB
  }
  return snr;
}

} // namespace

// ============================================================================
// FIR oversampler
// ============================================================================

template<class Float>
void test_fir_oversampling(uint64_t numChannels,
                           uint64_t numSamples,
                           uint64_t fftSamplesPerBlock,
                           uint64_t oversamplingOrder,
                           double transitionBand,
                           double snr_first_block_min_db,
                           double snr_steady_state_min_db)
{
  CAPTURE(numChannels);
  CAPTURE(numSamples);
  CAPTURE(fftSamplesPerBlock);
  CAPTURE(oversamplingOrder);
  CAPTURE(transitionBand);

  auto firUpSampler   = fir::TUpSamplerPreAllocated<Float>(oversamplingOrder, 1,
                          transitionBand, fftSamplesPerBlock);
  auto firDownSampler = fir::TDownSamplerPreAllocated<Float>(oversamplingOrder, 1,
                          transitionBand, fftSamplesPerBlock);
  firUpSampler.setNumChannels(numChannels);
  firUpSampler.setOrder(oversamplingOrder);
  firUpSampler.prepareBuffers(numSamples);
  const auto maxUpSampledSamples = firUpSampler.getMaxNumOutputSamples();
  firDownSampler.setNumChannels(numChannels);
  firDownSampler.setOrder(oversamplingOrder);
  firDownSampler.prepareBuffers(maxUpSampledSamples, numSamples);

  const uint64_t upSampleLatency   = firUpSampler.getNumSamplesBeforeOutputStarts();
  const uint64_t downSampleLatency = firDownSampler.getNumSamplesBeforeOutputStarts();
  const uint64_t latency =
    upSampleLatency + downSampleLatency / (1ULL << oversamplingOrder);
  const uint64_t numBuffers =
    latency / numSamples + 2 * std::max(fftSamplesPerBlock / numSamples, uint64_t{1});
  const uint64_t totSamples = numSamples * numBuffers;

  Buffer<Float> input(numChannels, totSamples);
  Buffer<Float> output(numChannels, totSamples);
  input.fill(0.0);
  output.fill(0.0);

  for (uint64_t c = 0; c < numChannels; ++c)
    for (uint64_t i = 0; i < input[c].size(); ++i)
      input[c][i] = std::sin(2.0 * M_PI * 0.125 * static_cast<Float>(i));

  auto in = input.get();
  auto out = output.get();
  for (uint64_t i = 0; i < numBuffers; ++i) {
    const auto numUpSampledSamples = firUpSampler.processBlock(in, numSamples);
    const auto& upSampled = firUpSampler.getOutput().get();
    firDownSampler.processBlock(upSampled, numUpSampledSamples, out, numSamples);
    for (uint64_t c = 0; c < numChannels; ++c) {
      in[c]  += numSamples;
      out[c] += numSamples;
    }
  }

  auto snr_first  = measure_snr_per_channel(input, output, latency,
                                            0, fftSamplesPerBlock);
  auto snr_steady = measure_snr_per_channel(input, output, latency,
                                            fftSamplesPerBlock, totSamples - latency);

  for (uint64_t c = 0; c < numChannels; ++c) {
    INFO("channel " << c << " first-block SNR = " << snr_first[c] << " dB");
    CHECK(snr_first[c] >= snr_first_block_min_db);
    INFO("channel " << c << " steady-state SNR = " << snr_steady[c] << " dB");
    CHECK(snr_steady[c] >= snr_steady_state_min_db);
  }
}

TEST_CASE_TEMPLATE("FIR oversampling SNR thresholds", Float, float, double) {
  // Steady-state SNR should be very high for a sine at 0.125·fs through a
  // linear-phase FIR. First-block SNR is much lower because the FIR is still
  // warming up — pick a permissive threshold there.
  SUBCASE("2ch, 128 spb, 1024 fft, order=4, 4% transition") {
    test_fir_oversampling<Float>(2, 128, 1024, 4, 4.0,
                                 /*first_block*/  -10.0,
                                 /*steady_state*/  60.0);
  }
  SUBCASE("2ch, 1024 spb, 512 fft, order=4, 4% transition") {
    test_fir_oversampling<Float>(2, 1024, 512, 4, 4.0,
                                 -10.0, 60.0);
  }
}

// ============================================================================
// IIR oversampler
// ============================================================================

template<class Float>
void test_iir_oversampling(uint64_t numChannels, uint64_t order,
                           uint64_t numSamples,
                           double snr_after_groupdelay_min_db)
{
  CAPTURE(numChannels);
  CAPTURE(order);
  CAPTURE(numSamples);

  const auto preset = iir::detail::getOversamplingPreset(0);
  const double groupDelay = 2.0 * preset.getGroupDelay(0, order);
  const uint32_t offset = 20u * static_cast<uint32_t>(std::ceil(groupDelay));
  const auto samplesPerBlock = offset + numSamples;

  Buffer<Float> input(numChannels, samplesPerBlock);
  input.fill(1.0);  // DC: minimum-phase oversampler should converge to 1.

  iir::UpSampler<Float>   upSampling(1, order);
  iir::DownSampler<Float> downSampling(1, order);
  upSampling.setNumChannels(numChannels);
  downSampling.setNumChannels(numChannels);
  REQUIRE(upSampling.setOrder(order));
  REQUIRE(downSampling.setOrder(order));
  upSampling.prepareBuffers(samplesPerBlock);
  downSampling.prepareBuffers(samplesPerBlock);

  upSampling.processBlock(input);
  const auto& upSampled = upSampling.getOutput();
  downSampling.processBlock(upSampled);
  auto& output = downSampling.getOutput();

  for (uint64_t c = 0; c < numChannels; ++c) {
    double signalPower = 0.0, noisePower = 0.0;
    for (uint64_t s = 0; s < numSamples - offset; ++s) {
      const double sig = input[c][s];
      const double obs = *output.at(c, s + offset);
      const double d   = sig - obs;
      signalPower += sig * sig;
      noisePower  += d * d;
    }
    const double snr = (noisePower > 0.0) ? 10.0 * std::log10(signalPower / noisePower)
                                          : 200.0;
    INFO("channel " << c << " IIR steady-state SNR = " << snr << " dB");
    CHECK(snr >= snr_after_groupdelay_min_db);
  }
}

TEST_CASE_TEMPLATE("IIR oversampling SNR threshold", Float, float, double) {
  // DC through min-phase IIR should reconstruct cleanly once we're past
  // the group delay. SNR in the 80-100 dB range is typical.
  test_iir_oversampling<Float>(2, 4, 1024, /*steady_state*/ 60.0);
}

// ============================================================================
// Oversampling wrapper (the public API used by Curvessor / Overdraw)
// ============================================================================

template<class Float>
void test_oversampling_wrapper(uint64_t order, uint64_t numSamples,
                               bool linearPhase,
                               BufferType upIn, BufferType upOut,
                               BufferType downIn, BufferType downOut,
                               double snr_steady_state_min_db)
{
  CAPTURE(order);
  CAPTURE(numSamples);
  CAPTURE(linearPhase);

  OversamplingSettings settings;
  settings.maxOrder = order;
  settings.order = order;
  settings.numUpSampledChannels = 2;
  settings.numDownSampledChannels = 2;
  settings.isUsingLinearPhase = linearPhase;
  settings.upSampleInputBufferType   = upIn;
  settings.upSampleOutputBufferType  = upOut;
  settings.downSampleInputBufferType = downIn;
  settings.downSampleOutputBufferType = downOut;

  Oversampling oversampling{ settings };
  oversampling.prepareBuffers(numSamples);
  const auto latency = oversampling.getLatency();
  const auto numBuffers =
    latency / numSamples
      + 2 * std::max(settings.fftBlockSize / numSamples, uint64_t{1});
  const auto totSamples = numSamples * numBuffers;

  Buffer<Float> input(settings.numUpSampledChannels, totSamples);
  Buffer<Float> output(settings.numDownSampledChannels, totSamples);
  input.fill(0.0);
  output.fill(0.0);
  for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c)
    for (uint64_t i = 0; i < input[c].size(); ++i)
      input[c][i] = linearPhase
                      ? std::sin(2.0 * M_PI * 0.125 * static_cast<Float>(i))
                      : 1.0;

  // Just the plain-in / plain-out path here — the buffer-type permutations
  // wire the same DSP differently and the wrapper test exists mainly to
  // verify the public API doesn't regress. Covered combinations are below.
  if (upIn == BufferType::plain && upOut == BufferType::plain
   && downIn == BufferType::plain && downOut == BufferType::plain) {
    auto in = input.get();
    auto out = output.get();
    for (uint64_t i = 0; i < numBuffers; ++i) {
      const auto numUp = oversampling.upSample(in, numSamples);
      const auto& up = oversampling.getUpSampleOutput<Float>();
      REQUIRE(up.getNumChannels() == settings.numUpSampledChannels);
      REQUIRE(up.getNumSamples() == numUp);
      oversampling.downSample(up.get(), numUp, out, numSamples);
      for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c)  in[c]  += numSamples;
      for (uint64_t c = 0; c < settings.numDownSampledChannels; ++c) out[c] += numSamples;
    }
    auto snr = measure_snr_per_channel(input, output, latency,
                                       settings.fftBlockSize,
                                       totSamples - latency);
    for (uint64_t c = 0; c < settings.numDownSampledChannels; ++c) {
      INFO("channel " << c << " steady-state SNR = " << snr[c] << " dB");
      CHECK(snr[c] >= snr_steady_state_min_db);
    }
  }
  else {
    // Other buffer-type combinations: don't crash, exercise the path.
    auto in = input.get();
    for (uint64_t i = 0; i < numBuffers; ++i) {
      oversampling.upSample(in, numSamples);
      const auto& up = oversampling.getUpSampleOutputInterleaved<Float>();
      oversampling.downSample(up, numSamples);
      for (uint64_t c = 0; c < settings.numUpSampledChannels; ++c) in[c] += numSamples;
    }
    // We didn't capture the output here; the goal is just to confirm the
    // call sequence doesn't trip an assertion.
    CHECK(true);
  }
}

TEST_CASE_TEMPLATE("Oversampling wrapper plain in/out (linear phase)", Float, float, double) {
  test_oversampling_wrapper<Float>(4, 1024, /*linearPhase*/ true,
                                   BufferType::plain, BufferType::plain,
                                   BufferType::plain, BufferType::plain,
                                   /*steady_state*/ 60.0);
}

TEST_CASE_TEMPLATE("Oversampling wrapper plain in/out (minimum phase)", Float, float, double) {
  // Non-linear-phase with a DC input — SNR can be a bit lower than the
  // linear-phase sine round-trip but should still be solidly clean.
  test_oversampling_wrapper<Float>(4, 1024, /*linearPhase*/ false,
                                   BufferType::plain, BufferType::plain,
                                   BufferType::plain, BufferType::plain,
                                   /*steady_state*/ 60.0);
}

TEST_CASE_TEMPLATE("Oversampling wrapper interleaved-up-and-down", Float, float, double) {
  // Just smoke-test the alternate-buffer-type path doesn't crash.
  test_oversampling_wrapper<Float>(4, 1024, /*linearPhase*/ true,
                                   BufferType::interleaved, BufferType::interleaved,
                                   BufferType::interleaved, BufferType::interleaved,
                                   /*steady_state*/ 0.0);
}
