from memory import Arc
from os.atomic import Atomic
from utils.lock import BlockingSpinLock, BlockingScopedLock

from CUDACompat import CUDAEventType, cudaEventDestroy, cudaGetDevice, cudaSetDevice


struct _EventCacheSlot(Movable):
    var device: Int
    var available: List[CUDAEventType]
    var lock: BlockingSpinLock
    var outstanding: Atomic[DType.int64]

    fn __init__(out self):
        self.device = -1
        self.available = List[CUDAEventType]()
        self.lock = BlockingSpinLock()
        self.outstanding = Atomic[DType.int64](0)

    fn __init__(out self, device: Int):
        self.device = device
        self.available = List[CUDAEventType]()
        self.lock = BlockingSpinLock()
        self.outstanding = Atomic[DType.int64](0)

    fn __moveinit__(out self, var other: Self):
        debug_assert(
            other.outstanding.load() == 0,
            "_EventCacheSlot moved while events are still outstanding",
        )
        self.device = other.device
        self.available = other.available^
        self.lock = BlockingSpinLock()
        self.outstanding = Atomic[DType.int64](0)

    fn tryToGet(mut self) -> Optional[CUDAEventType]:
        var event = CUDAEventType()
        with BlockingScopedLock(self.lock):
            if self.available.__len__() == 0:
                return None
            event = self.available.pop()
        _ = self.outstanding.fetch_add(1)
        return event^

    fn addNewOutstanding(mut self):
        _ = self.outstanding.fetch_add(1)

    fn _addBack(mut self, owned event: CUDAEventType):
        if event.event_id != 0:
            with BlockingScopedLock(self.lock):
                self.available.append(event^)
        _ = self.outstanding.fetch_sub(1)

    fn clear(mut self):
        debug_assert(
            self.outstanding.load() == 0,
            "_EventCacheSlot cleared while events are still outstanding",
        )
        while True:
            var event = CUDAEventType()
            with BlockingScopedLock(self.lock):
                if self.available.__len__() == 0:
                    break
                event = self.available.pop()
            self._destroyEvent(event)

    fn _destroyEvent(self, mut event: CUDAEventType):
        if event.event_id == 0:
            return

        var previous_device: Int = -1
        _ = cudaGetDevice(previous_device)
        if self.device != -1:
            _ = cudaSetDevice(self.device)
        _ = cudaEventDestroy(event)
        if previous_device != -1:
            _ = cudaSetDevice(previous_device)


# Owns one CUDA event and runs the EventCache deleter when the last Arc
# reference drops.
struct _EventOwner(Movable):
    var event: CUDAEventType
    var cache: UnsafePointer[_EventCacheSlot]

    fn __init__(out self, owned event: CUDAEventType):
        self.event = event^
        self.cache = UnsafePointer[_EventCacheSlot]()

    fn __init__(
        out self, owned event: CUDAEventType, cache: UnsafePointer[_EventCacheSlot]
    ):
        self.event = event^
        self.cache = cache

    fn __moveinit__(out self, var other: Self):
        self.event = other.event
        self.cache = other.cache
        other.event = CUDAEventType()
        other.cache = UnsafePointer[_EventCacheSlot]()

    fn __del__(var self):
        if self.event.event_id == 0:
            return
        if self.cache != UnsafePointer[_EventCacheSlot]():
            self.cache[]._addBack(self.event^)
            self.cache = UnsafePointer[_EventCacheSlot]()
            return
        _ = cudaEventDestroy(self.event)


# Equivalent to cms::cuda::SharedEventPtr.
# In C++: shared_ptr<remove_pointer_t<cudaEvent_t>>
# In Mojo: Arc handles the reference count; _EventOwner returns cached events
# to EventCache or destroys uncached events.
alias SharedEventPtr = Arc[_EventOwner]
