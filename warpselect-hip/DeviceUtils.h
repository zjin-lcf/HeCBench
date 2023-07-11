/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include "FaissAssert.h"
#include <vector>

namespace faiss {
namespace gpu {

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Starts the profiler (exposed via SWIG)
void profilerStart();

/// Stops the profiler (exposed via SWIG)
void profilerStop();

/// Synchronizes the CPU against all devices (equivalent to
/// hipDeviceSynchronize for each device)
void synchronizeAllDevices();

/// Returns a cached hipDeviceProp for the given device
const hipDeviceProp_t& getDeviceProperties(int device);

/// Returns the cached hipDeviceProp for the current device
const hipDeviceProp_t& getCurrentDeviceProperties();

/// Returns the maximum number of threads available for the given GPU
/// device
int getMaxThreads(int device);

/// Equivalent to getMaxThreads(getCurrentDevice())
int getMaxThreadsCurrentDevice();

/// Returns the maximum smem available for the given GPU device
size_t getMaxSharedMemPerBlock(int device);

/// Equivalent to getMaxSharedMemPerBlock(getCurrentDevice())
size_t getMaxSharedMemPerBlockCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Does the given device support full unified memory sharing host
/// memory?
bool getFullUnifiedMemSupport(int device);

/// Equivalent to getFullUnifiedMemSupport(getCurrentDevice())
bool getFullUnifiedMemSupportCurrentDevice();

/// Does the given device support tensor core operations?
bool getTensorCoreSupport(int device);

/// Equivalent to getTensorCoreSupport(getCurrentDevice())
bool getTensorCoreSupportCurrentDevice();

/// RAII object to set the current device, and restore the previous
/// device upon destruction
class DeviceScope {
   public:
    explicit DeviceScope(int device);
    ~DeviceScope();

   private:
    int prevDevice_;
};

// RAII object to manage a hipEvent_t
class HipEvent {
   public:
    /// Creates an event and records it in this stream
    explicit HipEvent(hipStream_t stream, bool timer = false);
    HipEvent(const HipEvent& event) = delete;
    HipEvent(HipEvent&& event) noexcept;
    ~HipEvent();

    inline hipEvent_t get() {
        return event_;
    }

    /// Wait on this event in this stream
    void streamWaitOnEvent(hipStream_t stream);

    /// Have the CPU wait for the completion of this event
    void cpuWaitOnEvent();

    HipEvent& operator=(HipEvent&& event) noexcept;
    HipEvent& operator=(HipEvent& event) = delete;

   private:
    hipEvent_t event_;
};

/// Wrapper to test return status of HIP functions
#define HIP_VERIFY(X)                      \
    do {                                    \
        auto err__ = (X);                   \
        FAISS_ASSERT_FMT(                   \
                err__ == hipSuccess,       \
                "HIP error %d %s",         \
                (int)err__,                 \
                hipGetErrorString(err__)); \
    } while (0)

/// Wrapper to synchronously probe for HIP errors
// #define FAISS_GPU_SYNC_ERROR 1

#ifdef FAISS_GPU_SYNC_ERROR
#define HIP_TEST_ERROR()                     \
    do {                                      \
        HIP_VERIFY(hipDeviceSynchronize()); \
    } while (0)
#else
#define HIP_TEST_ERROR()                \
    do {                                 \
        HIP_VERIFY(hipGetLastError()); \
    } while (0)
#endif

/// Call for a collection of streams to wait on
template <typename L1, typename L2>
void streamWaitBase(const L1& listWaiting, const L2& listWaitOn) {
    // For all the streams we are waiting on, create an event
    std::vector<hipEvent_t> events;
    for (auto& stream : listWaitOn) {
        hipEvent_t event;
        HIP_VERIFY(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        HIP_VERIFY(hipEventRecord(event, stream));
        events.push_back(event);
    }

    // For all the streams that are waiting, issue a wait
    for (auto& stream : listWaiting) {
        for (auto& event : events) {
            HIP_VERIFY(hipStreamWaitEvent(stream, event, 0));
        }
    }

    for (auto& event : events) {
        HIP_VERIFY(hipEventDestroy(event));
    }
}

/// These versions allow usage of initializer_list as arguments, since
/// otherwise {...} doesn't have a type
template <typename L1>
void streamWait(const L1& a, const std::initializer_list<hipStream_t>& b) {
    streamWaitBase(a, b);
}

template <typename L2>
void streamWait(const std::initializer_list<hipStream_t>& a, const L2& b) {
    streamWaitBase(a, b);
}

inline void streamWait(
        const std::initializer_list<hipStream_t>& a,
        const std::initializer_list<hipStream_t>& b) {
    streamWaitBase(a, b);
}

} // namespace gpu
} // namespace faiss
