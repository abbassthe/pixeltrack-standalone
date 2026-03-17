from CachingDeviceAllocator import CachingDeviceAllocator
from getCachingDeviceAllocator import getCachingDeviceAllocator
from allocator.deviceAllocatorStatus import GpuCachedBytes


# C++ shape: cms::cuda::deviceAllocatorStatus()
@always_inline
fn deviceAllocatorStatus() -> GpuCachedBytes:
    return getCachingDeviceAllocator().cacheStatus()


