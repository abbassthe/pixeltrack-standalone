from memory import OwnedPointer
from std.sys.info import size_of

from CUDACompat import cudaGetDevice, CUDAStreamType, cudaStreamDefault
from allocate_device import _AllocateDeviceState


alias cudaStream_t = CUDAStreamType


@always_inline
fn current_device() -> Int32:
    var device: Int = 0
    _ = cudaGetDevice(device)
    return Int32(device)


# Holds the raw device pointer and owns its lifetime.
struct _DeviceAllocation[T: AnyType](Movable, Defaultable):
    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    var device: Int32
    var stream: cudaStream_t
    var state: UnsafePointer[_AllocateDeviceState, MutAnyOrigin]

    @always_inline
    fn __init__(out self):
        self.ptr = UnsafePointer[Self.T, MutAnyOrigin]()
        self.device = Int32(-1)
        self.stream = cudaStreamDefault
        self.state = UnsafePointer[_AllocateDeviceState, MutAnyOrigin]()

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Self.T, MutAnyOrigin], device: Int32, stream: cudaStream_t, state: UnsafePointer[_AllocateDeviceState, MutAnyOrigin]):
        self.ptr = ptr
        self.device = device
        self.stream = stream
        self.state = state

    @always_inline
    fn __moveinit__(out self, var take: Self):
        self.ptr = take.ptr
        self.device = take.device
        self.stream = take.stream
        self.state = take.state
        take.ptr = UnsafePointer[Self.T, MutAnyOrigin]()
        take.device = Int32(-1)
        take.stream = cudaStreamDefault
        take.state = UnsafePointer[_AllocateDeviceState, MutAnyOrigin]()

    fn __del__(var self):
        if self.ptr != UnsafePointer[Self.T, MutAnyOrigin]() and self.device >= Int32(0) and self.state != UnsafePointer[_AllocateDeviceState, MutAnyOrigin]():
            self.state[].free_device(self.device, self.ptr.bitcast[UInt8](), self.stream)

    @always_inline
    fn get(self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        return self.ptr

    @always_inline
    fn release(mut self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        var released = self.ptr
        self.ptr = UnsafePointer[Self.T, MutAnyOrigin]()
        self.device = Int32(-1)
        self.stream = cudaStreamDefault
        self.state = UnsafePointer[_AllocateDeviceState, MutAnyOrigin]()
        return released

    @always_inline
    fn reset(
        mut self,
        ptr: UnsafePointer[Self.T, MutAnyOrigin] = UnsafePointer[Self.T, MutAnyOrigin](),
        device: Int32 = Int32(-1),
        stream: cudaStream_t = cudaStreamDefault,
    ):
        if self.ptr != UnsafePointer[Self.T, MutAnyOrigin]() and self.device >= Int32(0) and self.state != UnsafePointer[_AllocateDeviceState, MutAnyOrigin]():
            self.state[].free_device(self.device, self.ptr.bitcast[UInt8](), self.stream)
        self.ptr = ptr
        self.device = device
        self.stream = stream


# Equivalent to cms::cuda::device::unique_ptr<T>
alias unique_ptr[T: AnyType] = OwnedPointer[_DeviceAllocation[T]]


fn make_device_unique[T: AnyType](mut state: _AllocateDeviceState, stream: cudaStream_t = cudaStreamDefault) raises -> unique_ptr[T]:
    var dev = current_device()
    var mem = state.allocate_device(dev, UInt(size_of[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream, UnsafePointer(to=state)))


fn make_device_unique[T: AnyType](
    n: UInt,
    mut state: _AllocateDeviceState,
    stream: cudaStream_t = cudaStreamDefault,
) raises -> unique_ptr[T]:
    var dev = current_device()
    var mem = state.allocate_device(dev, n * UInt(size_of[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream, UnsafePointer(to=state)))


fn make_device_unique_uninitialized[T: AnyType](
    mut state: _AllocateDeviceState,
    stream: cudaStream_t = cudaStreamDefault,
) raises -> unique_ptr[T]:
    var dev = current_device()
    var mem = state.allocate_device(dev, UInt(size_of[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream, UnsafePointer(to=state)))


fn make_device_unique_uninitialized[T: AnyType](
    n: UInt,
    mut state: _AllocateDeviceState,
    stream: cudaStream_t = cudaStreamDefault,
) raises -> unique_ptr[T]:
    var dev = current_device()
    var mem = state.allocate_device(dev, n * UInt(size_of[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream, UnsafePointer(to=state)))
