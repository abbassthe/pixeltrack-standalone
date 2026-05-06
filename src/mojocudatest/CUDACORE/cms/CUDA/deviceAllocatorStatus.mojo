from getCachingDeviceAllocator import _CachingAllocatorState
from allocator.deviceAllocatorStatus import GpuCachedBytes


# C++ shape: cms::cuda::deviceAllocatorStatus()
@always_inline
fn deviceAllocatorStatus(mut state: _CachingAllocatorState) -> GpuCachedBytes:
    return state.getCachingDeviceAllocator().cacheStatus()


