# Mojo port of CUDACore/getCachingHostAllocator.h's HostTraits.
#
# This file used to hold a ~690-line transcription of the notcub
# CachingHostAllocator -- a near-duplicate of CachingDeviceAllocator with
# cudaMallocHost swapped in. That was a port of the `cuda` backend's dead code:
# `cudadev` still ships notcub/CachingHostAllocator.h but nothing includes it,
# having replaced it with GenericCachingAllocator<HostTraits>.
#
# Worse, the duplicate carried the *device* reuse policy, which is wrong for
# pinned host memory. See HostTraits below for the three places they differ.
# The shared implementation now lives in CachingDeviceAllocator.mojo,
# parameterised by CachingAllocatorTraits.
from MojoCudaDev.CUDACore.CachingDeviceAllocator import (
    CachingDeviceAllocator,
    CachingAllocatorTraits,
)
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState


# C++: struct HostTraits -- getCachingHostAllocator.h.
struct HostTraits(CachingAllocatorTraits):
    # C++ reaches the global runtime via cudaHostAlloc/cudaFreeHost; this port
    # carries the state explicitly (allocate_host.mojo:17).
    alias State = _AllocateHostState

    # C++: kHostDevice = 0 -- pinned memory is not owned by any one device.
    comptime kHostDevice: Int = 0

    @staticmethod
    fn allocate(
        mut state: Self.State,
        device: Int,
        nbytes: UInt,
        stream: CUDAStreamType,
    ) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
        # device is deliberately unused: pinned memory is device-independent
        return state.allocate_host_raw(nbytes, stream)

    @staticmethod
    fn free(
        mut state: Self.State,
        device: Int,
        ptr: UnsafePointer[UInt8, MutAnyOrigin],
        stream: CUDAStreamType,
    ):
        state.free_host_raw(ptr)

    # --- the three policy differences from DeviceTraits ---

    # C++: `return compare();` -- device is ignored entirely, so the cached set
    # is ordered by size alone. The device version sorts by device first, which
    # would fragment one shared pinned pool into per-device pools that cannot
    # borrow from each other.
    @staticmethod
    fn device_compare(a_dev: Int, b_dev: Int, tie: Bool) -> Bool:
        return tie

    # C++: `return kHostDevice;` -- memory always comes from the host, whatever
    # device the event was recorded on.
    @staticmethod
    fn memory_device(event_device: Int) -> Int:
        return Self.kHostDevice

    # C++: `return true;` -- a block freed while device 1 was current is still
    # perfectly usable for device 0.
    @staticmethod
    fn can_reuse_in_device(cached_device: Int, want_device: Int) -> Bool:
        return True

    # C++: `return false;` -- for device memory the stream orders same-stream
    # reuse, so it is safe without checking the event. That reasoning does not
    # hold for pinned host memory, so this always falls through to the event
    # check instead.
    @staticmethod
    fn can_reuse_in_queue(a: CUDAStreamType, b: CUDAStreamType) -> Bool:
        return False


# C++: using CachingHostAllocator = GenericCachingAllocator<HostTraits>;
comptime CachingHostAllocator = CachingDeviceAllocator[HostTraits]