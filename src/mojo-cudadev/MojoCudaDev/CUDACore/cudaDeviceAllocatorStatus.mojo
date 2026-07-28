from MojoCudaDev.CUDACore.getCachingDeviceAllocator import _CachingAllocatorState
from MojoCudaDev.CUDACore.CachingDeviceAllocator import GpuCachedBytes


# C++ shape: cms::cuda::deviceAllocatorStatus()
@always_inline
fn deviceAllocatorStatus(mut state: _CachingAllocatorState) -> GpuCachedBytes:
    return state.getCachingDeviceAllocator()[].CacheStatus()


