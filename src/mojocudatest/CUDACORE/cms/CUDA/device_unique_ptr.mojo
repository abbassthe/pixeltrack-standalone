from memory import OwnedPointer

from CUDACompat import cudaGetDevice, CUDAStreamType, cudaStreamDefault
from allocate_device import allocate_device, free_device


alias cudaStream_t = CUDAStreamType


@always_inline
fn current_device() -> Int32:
    var device: Int = 0
    _ = cudaGetDevice(device)
    return Int32(device)


# Holds the raw device pointer and owns its lifetime.
# OwnedPointer[_DeviceAllocation[T]] calls __del__ on destruction,
# giving us the unique_ptr<T, DeviceDeleter> semantics from C++.
struct _DeviceAllocation[T](Movable, Defaultable):
    var ptr: UnsafePointer[T]
    var device: Int32
    var stream: cudaStream_t

    @always_inline
    fn __init__(out self):
        self.ptr = UnsafePointer[T]()
        self.device = Int32(-1)
        self.stream = cudaStreamDefault

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[T], device: Int32, stream: cudaStream_t):
        self.ptr = ptr
        self.device = device
        self.stream = stream

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self.ptr = other.ptr
        self.device = other.device
        self.stream = other.stream
        other.ptr = UnsafePointer[T]()
        other.device = Int32(-1)
        other.stream = cudaStreamDefault

    fn __del__(var self):
        if self.ptr != UnsafePointer[T]() and self.device >= Int32(0):
            free_device(self.device, self.ptr.bitcast[UInt8](), self.stream)

    @always_inline
    fn get(self) -> UnsafePointer[T]:
        return self.ptr

    @always_inline
    fn release(mut self) -> UnsafePointer[T]:
        var released = self.ptr
        self.ptr = UnsafePointer[T]()
        self.device = Int32(-1)
        self.stream = cudaStreamDefault
        return released

    @always_inline
    fn reset(
        mut self,
        ptr: UnsafePointer[T] = UnsafePointer[T](),
        device: Int32 = Int32(-1),
        stream: cudaStream_t = cudaStreamDefault,
    ):
        if self.ptr != UnsafePointer[T]() and self.device >= Int32(0):
            free_device(self.device, self.ptr.bitcast[UInt8](), self.stream)
        self.ptr = ptr
        self.device = device
        self.stream = stream


# Equivalent to cms::cuda::device::unique_ptr<T>
alias unique_ptr[T] = OwnedPointer[_DeviceAllocation[T]]


fn make_device_unique[T](stream: cudaStream_t = cudaStreamDefault) -> unique_ptr[T] raises:
    let dev = current_device()
    let mem = allocate_device(dev, UInt(sizeof[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream))


fn make_device_unique[T](
    n: UInt,
    stream: cudaStream_t = cudaStreamDefault,
) -> unique_ptr[T] raises:
    let dev = current_device()
    let mem = allocate_device(dev, n * UInt(sizeof[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream))


fn make_device_unique_uninitialized[T](
    stream: cudaStream_t = cudaStreamDefault,
) -> unique_ptr[T] raises:
    let dev = current_device()
    let mem = allocate_device(dev, UInt(sizeof[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream))


fn make_device_unique_uninitialized[T](
    n: UInt,
    stream: cudaStream_t = cudaStreamDefault,
) -> unique_ptr[T] raises:
    let dev = current_device()
    let mem = allocate_device(dev, n * UInt(sizeof[T]()), stream)
    return unique_ptr[T](_DeviceAllocation[T](mem.bitcast[T](), dev, stream))
