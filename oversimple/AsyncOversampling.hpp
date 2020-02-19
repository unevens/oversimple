/*
Copyright 2020 Dario Mambro

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
#include "../lockfree-async/LockFreeAsync.hpp"
#include "oversimple/Oversampling.hpp"

namespace oversimple {

/**
 * Using lockfree::Async you can control an Oversampling instance through an
 * OversamplingSettings object asyncrhonously. In this way, all the memory
 * allocations and computations needed to setup the oversampling objects will be
 * executed on their own thread.
 *
 * Usign an OversamplingGetter object, you access the Oversampling functionality
 * on realtime threads.
 *
 * The lockfree::Async class takes care of preventing locks and data races.
 *
 * @see Oversampling
 * @see OversamplingSettings
 * @see OversamplingGetter
 */

using AsyncOversampling = lockfree::Async<OversamplingSettings>;

/**
 * A OversamplingGetter object lets you access the Oversampling functionality
 * on realtime threads.
 *
 * @see Oversampling
 * @see OversamplingSettings
 */
template<typename Scalar>
using OversamplingGetter =
  lockfree::Async<OversamplingSettings>::Getter<Oversampling<Scalar>>;

/**
 * Function to request an OversamplingGetter from an AsyncOversampling instance.
 * @see AsyncOversampling
 * @see OversamplingGetter
 */
template<typename Scalar>
inline OversamplingGetter<Scalar>*
RequestOversamplingGetter(AsyncOversampling& asyncOversampling)
{
  return asyncOversampling.requestGetter<Oversampling<Scalar>>();
}

/**
 * A getter for the OversamplingSettings.
 */
using OversamplingSettingsGetter =
  lockfree::Async<OversamplingSettings>::BlockingGetter<>;

/**
 * Function to request an OversamplingSettingsGetter from an AsyncOversampling
 * instance.
 * @see AsyncOversampling
 * @see OversamplingSettingsGetter
 */
inline OversamplingSettingsGetter*
RequestOversamplingSettingsGetter(AsyncOversampling& asyncOversampling)
{
  return asyncOversampling.requestBlockingGetter<>();
}

} // namespace oversimple
