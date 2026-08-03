# Mojo port of CUDACore/ESProduct.h.
#
# Per-GPU-device lazy-fill cache for EventSetup/conditions data: at most one
# thread transfers data to a given device; other threads either get the
# already-filled data immediately (lock-free fast path via `filled`), or wait
# on the CUDA event that marks the transfer's completion (so they don't
# re-transfer, and so subsequent work on their own stream is correctly
# ordered after the transfer).
#
# C++'s `getEventCache()` (the global backing this) doesn't exist here --
# every "formerly-global" resource in this port is threaded through
# explicitly (see CUDAAppContext), so ESProduct's constructor takes the
# EventCache + CUDARuntime it needs instead of reaching for a global.
#
# `Item.filling_stream`'s "no one is filling" sentinel is `cudaStreamDefault`
# (`CUDAStreamType()`), preserving the same ambiguity C++'s `nullptr` has:
# CUDA's own default-stream handle *is* the null pointer value. So if the
# first filler happens to use the default stream, a second thread on a
# different real stream would also see "not filling yet" and could
# re-trigger transferAsync. This is a latent landmine already present in the
# C++ original, not introduced by the port -- left as-is rather than
# silently changed into different behavior.
#
# `Item` is heap-allocated per device (one alloc[Item[T]](1) per device,
# stored as a raw pointer) rather than held inline in a List, matching
# EventCache._buildSlots -- BlockingSpinLock can't live inline in a List
# (non-movable/non-copyable, the same constraint that showed up earlier in
# CachingHostAllocator).
from os.atomic import Atomic
from utils.lock import BlockingSpinLock, BlockingScopedLock

from MojoCudaDev.CUDACore.CUDACompat import (
    CUDAStreamType,
    cudaStreamDefault,
    CUDARuntime,
    cudaEventRecord,
    cudaStreamWaitEvent,
)
from MojoCudaDev.CUDACore.SharedEventPtr import SharedEventPtr
from MojoCudaDev.CUDACore.EventCache import EventCache
from MojoCudaDev.CUDACore.ScopedSetDevice import ScopedSetDevice
from MojoCudaDev.CUDACore.currentDevice import currentDevice
from MojoCudaDev.CUDACore.deviceCount import deviceCount
from MojoCudaDev.CUDACore.eventWorkHasCompleted import event_work_has_completed


# C++: private struct Item (ESProduct.h:95-102)
struct Item[T: Defaultable & Movable](Movable):
    var mutex: BlockingSpinLock
    var event: SharedEventPtr
    var filling_stream: CUDAStreamType
    var filled: Atomic[DType.uint8]
    var data: Self.T

    fn __init__(out self, var event: SharedEventPtr):
        self.mutex = BlockingSpinLock()
        self.event = event^
        self.filling_stream = cudaStreamDefault
        self.filled = Atomic[DType.uint8](0)
        self.data = Self.T()

    fn __moveinit__(out self, deinit take: Self):
        self.mutex = BlockingSpinLock()
        self.event = take.event^
        self.filling_stream = take.filling_stream
        self.filled = Atomic[DType.uint8](0)
        self.data = take.data^


struct ESProduct[T: Defaultable & Movable](Movable):
    var gpu_data_per_device: List[UnsafePointer[Item[Self.T], MutAnyOrigin]]

    # C++: ESProduct() : gpuDataPerDevice_(deviceCount()) { ... } (ESProduct.h:21-29)
    fn __init__(out self, mut event_cache: EventCache, mut runtime: CUDARuntime) raises:
        self.gpu_data_per_device = List[UnsafePointer[Item[Self.T], MutAnyOrigin]]()
        var n = deviceCount()
        for i in range(n):
            var guard = ScopedSetDevice(i)
            var event = event_cache.get(runtime)
            var slot = alloc[Item[Self.T]](1)
            __get_address_as_uninit_lvalue(slot.address) = Item[Self.T](event^)
            self.gpu_data_per_device.append(slot)

    fn __moveinit__(out self, deinit take: Self):
        self.gpu_data_per_device = take.gpu_data_per_device^

    # C++: ~ESProduct() = default; (ESProduct.h:31) -- the vector<Item>'s own
    # destructor releases each Item's SharedEventPtr/data; here that means
    # explicitly destroying and freeing each heap-allocated slot.
    fn __del__(deinit self):
        for i in range(self.gpu_data_per_device.__len__()):
            var slot = self.gpu_data_per_device[i]
            slot.destroy_pointee()
            slot.free()

    # C++: dataForCurrentDeviceAsync (ESProduct.h:36-92)
    fn dataForCurrentDeviceAsync(
        mut self,
        stream: CUDAStreamType,
        transferAsync: fn (mut Self.T, CUDAStreamType) raises -> None,
    ) raises -> UnsafePointer[Self.T, MutAnyOrigin]:
        var device = currentDevice()
        var slot = self.gpu_data_per_device[device]

        if slot[].filled.load() == 0:
            with BlockingScopedLock(slot[].mutex):
                if slot[].filled.load() == 0:
                    if slot[].filling_stream != cudaStreamDefault:
                        # Someone else is filling.
                        if event_work_has_completed(slot[].event[].event):
                            _ = slot[].filled.fetch_add(1)
                            slot[].filling_stream = cudaStreamDefault
                        elif slot[].filling_stream != stream:
                            _ = cudaStreamWaitEvent(stream, slot[].event[].event, 0)
                        # else: filling on the same stream -- nothing to do,
                        # subsequent work on this stream is already ordered
                        # after it.
                    else:
                        # Not yet on the GPU, and we're the first to try.
                        transferAsync(slot[].data, stream)
                        slot[].filling_stream = stream
                        _ = cudaEventRecord(slot[].event[].event, stream)

        return UnsafePointer(to=slot[].data)
