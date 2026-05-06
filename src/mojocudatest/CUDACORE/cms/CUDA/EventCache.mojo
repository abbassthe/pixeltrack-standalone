from CUDACompat import CUDAEventType, CUDARuntime, cudaEventCreateWithFlags
from MojoBridge.DTypes import cudaEventDisableTiming
from SharedEventPtr import SharedEventPtr, _EventCacheSlot, _EventOwner
from currentDevice import currentDevice
from cudaCheck import cudaCheck_
from deviceCount import deviceCount
from eventWorkHasCompleted import event_work_has_completed


struct EventCache(Movable):
    var cache_: List[UnsafePointer[_EventCacheSlot]]

    fn __init__(out self):
        self.cache_ = List[UnsafePointer[_EventCacheSlot]]()
        self._buildSlots()

    fn __moveinit__(out self, var other: Self):
        self.cache_ = other.cache_^
        other.cache_ = List[UnsafePointer[_EventCacheSlot]]()

    fn __del__(owned self):
        self._destroySlots()

    fn get(mut self, mut runtime: CUDARuntime) raises -> SharedEventPtr:
        let dev = currentDevice()
        var event = self.makeOrGet(dev, runtime)
        if event_work_has_completed(event[].event):
            return event^

        # Keep incomplete events alive while searching, otherwise the same
        # incomplete event can be returned to the cache and picked again.
        var incomplete = List[SharedEventPtr]()
        incomplete.append(event^)
        while True:
            event = self.makeOrGet(dev, runtime)
            if event_work_has_completed(event[].event):
                return event^
            incomplete.append(event^)

    fn makeOrGet(mut self, dev: Int, mut runtime: CUDARuntime) raises -> SharedEventPtr:
        var slot = self.cache_[dev]
        var cached = slot[].tryToGet()
        if cached:
            var event = cached.value()
            return SharedEventPtr(_EventOwner(event^, slot))

        var event = CUDAEventType()
        _ = cudaCheck_(
            __source_location().file_name(),
            __source_location().line(),
            "cudaEventCreateWithFlags",
            cudaEventCreateWithFlags(event, cudaEventDisableTiming, runtime),
        )
        slot[].addNewOutstanding()
        return SharedEventPtr(_EventOwner(event^, slot))

    fn clear(mut self):
        self._destroySlots()
        self._buildSlots()

    fn _buildSlots(mut self):
        for dev in range(deviceCount()):
            var slot = UnsafePointer[_EventCacheSlot].alloc(1)
            slot.init_pointee_move(_EventCacheSlot(dev))
            self.cache_.append(slot)

    fn _destroySlots(mut self):
        for i in range(self.cache_.__len__()):
            var slot = self.cache_[i]
            slot[].clear()
            slot.destroy_pointee()
            slot.free()
        self.cache_.clear()
