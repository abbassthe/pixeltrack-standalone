from MojoCudaDev.CUDACore.CachingHostAllocator import CachingHostAllocator
from MojoCudaDev.CUDACore.getCachingDeviceAllocator import (
    binGrowth,
    minBin,
    maxBin,
    debug,
    minCachedBytes,
    _formatBinSize,
)
from utils.lock import BlockingSpinLock, BlockingScopedLock


# C++ (getCachingHostAllocator.h) has no Policy / _selectPolicy() of its own --
# host allocation always goes through the one CachingHostAllocator, no
# synchronous/asynchronous/caching switch like the device side has.
struct _CachingHostAllocatorState(Movable):
    var _allocator_lock: BlockingSpinLock
    var _allocator_initialized: Bool
    var _allocator_ptr: UnsafePointer[CachingHostAllocator, MutAnyOrigin]

    fn __init__(out self):
        self._allocator_lock = BlockingSpinLock()
        self._allocator_initialized = False
        self._allocator_ptr = UnsafePointer[CachingHostAllocator, MutAnyOrigin]()

    fn __moveinit__(out self, var take: Self):
        self._allocator_lock = BlockingSpinLock()
        self._allocator_initialized = take._allocator_initialized
        self._allocator_ptr = take._allocator_ptr

    fn __del__(deinit self):
        if self._allocator_initialized and self._allocator_ptr != UnsafePointer[CachingHostAllocator, MutAnyOrigin]():
            self._allocator_ptr.destroy_pointee()
            self._allocator_ptr.free()

    # C++: inline CachingHostAllocator& getCachingHostAllocator() (getCachingHostAllocator.h:100-125)
    fn getCachingHostAllocator(mut self) -> UnsafePointer[CachingHostAllocator, MutAnyOrigin]:
        with BlockingScopedLock(self._allocator_lock):
            if not self._allocator_initialized:
                if debug:
                    _printHostSettings()
                self._allocator_ptr = alloc[CachingHostAllocator](1)
                __get_address_as_uninit_lvalue(self._allocator_ptr.address) = CachingHostAllocator(
                    binGrowth,
                    minBin,
                    maxBin,
                    minCachedBytes(),
                    False,  # do not skip cleanup
                    debug
                )
                self._allocator_initialized = True
            return self._allocator_ptr


# C++: the `if (debug) { std::cout << ... }` block inlined directly inside
# getCachingHostAllocator() (getCachingHostAllocator.h:101-120) -- factored out
# to a free function here, matching how the device file already factored its
# own copy of the same block into _printSettings().
fn _printHostSettings():
    print("cub::CachingHostAllocator settings")
    print("  bin growth " + String(binGrowth))
    print("  min bin    " + String(minBin))
    print("  max bin    " + String(maxBin))
    print("  resulting bins:")
    var bin = minBin
    while bin <= maxBin:
        var bin_size = CachingHostAllocator.IntPow(binGrowth, bin)
        print("    " + _formatBinSize(bin_size))
        bin += 1
    print(
        "  maximum amount of cached memory: "
        + String(minCachedBytes() >> 20)
        + " MB"
    )
