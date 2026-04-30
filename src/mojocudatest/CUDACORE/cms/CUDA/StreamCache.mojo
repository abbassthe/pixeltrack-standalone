from gpu.host import DeviceContext

from CUDACompat import CUDAStreamType, cudaGetDevice
from SharedStreamPtr import SharedStreamPtr
from deviceCount import deviceCount
from Framework.ReusableObjectHolder import (
    ReusableObjectHolder,
    ReusableObjectMaker,
    makeOrGet,
)


struct StreamMaker(ReusableObjectMaker, Movable):
    alias Output = CUDAStreamType

    var dev: Int

    fn __init__(out self, dev: Int):
        self.dev = dev

    fn make(mut self) raises -> CUDAStreamType:
        var ctx = DeviceContext(device_id=self.dev)
        # Largest priority is analogous to cudaStreamNonBlocking:
        # the stream does not implicitly synchronize with the default stream.
        return ctx.create_stream(priority=ctx.stream_priority_range().largest)


# Gets a (cached) CUDA stream for the current device. The stream is returned
# to the cache when the SharedStreamPtr reference count drops to zero.
# This function is thread safe.
struct StreamCache(Movable):
    var cache_: List[ReusableObjectHolder[CUDAStreamType]]

    fn __init__(out self):
        self.cache_ = List[ReusableObjectHolder[CUDAStreamType]]()
        for _ in range(deviceCount()):
            self.cache_.append(ReusableObjectHolder[CUDAStreamType]())

    fn __moveinit__(out self, var other: Self):
        self.cache_ = other.cache_^

    fn get(mut self) raises -> SharedStreamPtr:
        var dev: Int = 0
        _ = cudaGetDevice(dev)
        var maker = StreamMaker(dev)
        return makeOrGet[StreamMaker](self.cache_[dev], maker)

    # Not thread safe — intended to be called only from CUDAService destructor.
    fn clear(mut self):
        self.cache_.clear()
        for _ in range(deviceCount()):
            self.cache_.append(ReusableObjectHolder[CUDAStreamType]())


# Gets the global instance of a StreamCache.
# This function is thread safe.
var _global_stream_cache = StreamCache()

fn getStreamCache() -> ref [_global_stream_cache] StreamCache:
    return _global_stream_cache
