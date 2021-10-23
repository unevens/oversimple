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

#include "oversimple/IirOversamplingDesigner.hpp"

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

namespace oversimple::iir::detail {
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
inline double getOversamplingMinGroupDelay(uint32_t order, int presetIndex = 0)
{
  if (order == 0) {
    return 0.0;
  }
  return getOversamplingPreset(presetIndex).getMinGroupDelay(order);
}

} // namespace oversimple::iir::detail
