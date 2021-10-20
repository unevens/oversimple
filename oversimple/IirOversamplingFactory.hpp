/*
Copyright 2019 Dario Mambro

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

#include "oversimple/IirOversampling.hpp"

namespace hiir {
struct FakeInterface
{
  void clear_buffers() {}
  void set_coefs(const double coef_arr[]) {}
  void process_block(double out_ptr[], const double in_ptr[], long nbr_spl) {}
};
} // namespace hiir

#if AVEC_AVX512
#include "hiir/Downsampler2x16Avx512.h"
#include "hiir/Downsampler2x8F64Avx512.h"
#include "hiir/Upsampler2x16Avx512.h"
#include "hiir/Upsampler2x8F64Avx512.h"
#else
namespace hiir {
template<int NC>
class Downsampler2x8F64Avx final : public FakeInterface
{};
template<int NC>
class Downsampler2x16Avx final : public FakeInterface
{};
template<int NC>
class Upsampler2x8F64Avx final : public FakeInterface
{};
template<int NC>
class Upsampler2x16Avx final : public FakeInterface
{};
} // namespace hiir
#endif

#if AVEC_AVX
#include "hiir/Downsampler2x4F64Avx.h"
#include "hiir/Downsampler2x8Avx.h"
#include "hiir/Upsampler2x4F64Avx.h"
#include "hiir/Upsampler2x8Avx.h"
#else
namespace hiir {
template<int NC>
class Downsampler2x4F64Avx final : public FakeInterface
{};
template<int NC>
class Downsampler2x8Avx final : public FakeInterface
{};
template<int NC>
class Upsampler2x4F64Avx final : public FakeInterface
{};
template<int NC>
class Upsampler2x8Avx final : public FakeInterface
{};
} // namespace hiir
#endif

#if AVEC_SSE2
#include "hiir/Downsampler2x2F64Sse2.h"
#include "hiir/Downsampler2x4Sse.h"
#include "hiir/Upsampler2x2F64Sse2.h"
#include "hiir/Upsampler2x4Sse.h"
#else
namespace hiir {
template<int NC>
class Downsampler2x2F64Sse2 final : public FakeInterface
{};
template<int NC>
class Downsampler2x4Sse final : public FakeInterface
{};
template<int NC>
class Upsampler2x2F64Sse2 final : public FakeInterface
{};
template<int NC>
class Upsampler2x4Sse final : public FakeInterface
{};
} // namespace hiir
#endif

#if AVEC_NEON_64
#include "hiir/Downsampler2x2F64Neon.h"
#include "hiir/Upsampler2x2F64Neon.h"
#else
namespace hiir {
template<int NC>
class Downsampler2x2F64Neon final : public FakeInterface
{};
template<int NC>
class Upsampler2x2F64Neon final : public FakeInterface
{};
} // namespace hiir
#endif

#if AVEC_NEON
#include "hiir/Downsampler2x4Neon.h"
#include "hiir/Upsampler2x4Neon.h"
#else
namespace hiir {
template<int NC>
class Downsampler2x4Neon final : public FakeInterface
{};
template<int NC>
class Upsampler2x4Neon final : public FakeInterface
{};
} // namespace hiir
#endif

namespace oversimple::iir {
namespace detail {
/**
 * Returns an OversamplingDesigner object implementing a quality preset for
 *  Oversampling.
 * @param presetIndex an index identifying the preset
 * @return the OversamplingDesigner corresponding to the index
 */
inline OversamplingDesigner getOversamplingPreset(int presetIndex = 0)
{
  switch (presetIndex) {
    case 0:
    default:
      return { 140.0, 0.0443 };
    case 1:
      return { 142.0, 0.0464 };
  }
}

/**
 * Returns the minimum grup delay of the IIR antialiasing filters used when
 * oversampling with a specific oversampling order and quality preset.
 * @param order the oversampling order
 * @param presetIndex the oversampling quality preset
 * @return the minimum group delay
 */
inline double getOversamplingMinGroupDelay(int order, int presetIndex = 0)
{
  if (order == 0) {
    return 0.0;
  }
  return getOversamplingPreset(presetIndex).getMinGroupDelay(order);
}

/**
 * Static class implementing a factory for UpSampler.
 */
template<typename Scalar>
class UpSamplerFactory final
{
  template<int NC>
  using FakeUpsamplerStage8Double = hiir::FakeInterface;
  template<int NC>
  using FakeUpsamplerStage2Float = hiir::FakeInterface;

#if AVEC_X86

public:
  template<int NC>
  using Stage8 = typename std::
    conditional<std::is_same<Scalar, float>::value, hiir::Upsampler2x8Avx<NC>, FakeUpsamplerStage8Double<NC>>::type;
  template<int NC>
  using Stage4 = typename std::
    conditional<std::is_same<Scalar, float>::value, hiir::Upsampler2x4Sse<NC>, hiir::Upsampler2x4F64Avx<NC>>::type;
  template<int NC>
  using Stage2 = typename std::
    conditional<std::is_same<Scalar, float>::value, FakeUpsamplerStage2Float<NC>, hiir::Upsampler2x2F64Sse2<NC>>::type;

#elif AVEC_ARM

private:
  template<int NC>
  using FakeUpsamplerStage4Double = hiir::Upsampler2x4F64Avx<NC>;

public:
  template<int NC>
  using Stage8 = FakeUpsamplerStage8Double<NC>;
  template<int NC>
  using Stage4 = typename std::
    conditional<std::is_same<Scalar, float>::value, hiir::Upsampler2x4Neon<NC>, FakeUpsamplerStage4Double<NC>>::type;
  template<int NC>
  using Stage2 = typename std::
    conditional<std::is_same<Scalar, float>::value, FakeUpsamplerStage2Float<NC>, hiir::Upsampler2x2F64Neon<NC>>::type;

#else

  static_assert(false, "Unknown platform");

#endif

public:

  using UpSampler = TUpSampler<Scalar, 11, 5, 3, 3, 2, Stage8, Stage4, Stage2>;
};

/**
 * Static class implementing a factory for DownSamplers.
 */
template<typename Scalar>
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
    conditional<std::is_same<Scalar, float>::value, hiir::Downsampler2x8Avx<NC>, FakeDownsamplerStage8Double<NC>>::type;

  template<int NC>
  using Stage4 = typename std::
    conditional<std::is_same<Scalar, float>::value, hiir::Downsampler2x4Sse<NC>, hiir::Downsampler2x4F64Avx<NC>>::type;

  template<int NC>
  using Stage2 = typename std::conditional<std::is_same<Scalar, float>::value,
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
  using Stage4 = typename std::conditional<std::is_same<Scalar, float>::value,
                                           hiir::Downsampler2x4Neon<NC>,
                                           FakeDownsamplerStage4Double<NC>>::type;
  template<int NC>
  using Stage2 = typename std::conditional<std::is_same<Scalar, float>::value,
                                           FakeDownsamplerStage2Float<NC>,
                                           hiir::Downsampler2x2F64Neon<NC>>::type;

#else

  static_assert(false, "Unknown platform");

#endif

public:
  using DownSampler = TDownSampler<Scalar, 11, 5, 3, 3, 2, Stage8, Stage4, Stage2>;
};
} // namespace detail

template<typename Scalar>
class DownSampler final : public detail::DownSamplerFactory<Scalar>::DownSampler
{
public:
  explicit DownSampler(int numChannels, int orderToPreallocateFor = 0)
    : detail::DownSamplerFactory<Scalar>::DownSampler(detail::getOversamplingPreset(0),
                                                      numChannels,
                                                      orderToPreallocateFor)
  {}
};

template<typename Scalar>
class UpSampler final : public detail::UpSamplerFactory<Scalar>::UpSampler
{
public:
  explicit UpSampler(int numChannels, int orderToPreallocateFor = 0)
    : detail::UpSamplerFactory<Scalar>::UpSampler(detail::getOversamplingPreset(0), numChannels, orderToPreallocateFor)
  {}
};

} // namespace oversimple::iir
