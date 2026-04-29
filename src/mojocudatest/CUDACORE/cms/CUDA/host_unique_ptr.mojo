from memory import OwnedPointer

from CUDACompat import CUDAStreamType, cudaStreamDefault
from allocate_host import allocate_host, free_host


alias cudaStream_t = CUDAStreamType


# Holds the raw pinned-host pointer and owns its lifetime.
# OwnedPointer[_HostAllocation[T]] calls __del__ on destruction,
# giving us the unique_ptr<T, HostDeleter> semantics from C++.
struct _HostAllocation[T](Movable, Defaultable):
    var ptr: UnsafePointer[T]

    @always_inline
    fn __init__(out self):
        self.ptr = UnsafePointer[T]()

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[T]):
        self.ptr = ptr

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self.ptr = other.ptr
        other.ptr = UnsafePointer[T]()

    fn __del__(var self):
        if self.ptr != UnsafePointer[T]():
            free_host(self.ptr.bitcast[UInt8]())

    @always_inline
    fn get(self) -> UnsafePointer[T]:
        return self.ptr

    @always_inline
    fn release(mut self) -> UnsafePointer[T]:
        var released = self.ptr
        self.ptr = UnsafePointer[T]()
        return released

    @always_inline
    fn reset(mut self, ptr: UnsafePointer[T] = UnsafePointer[T]()):
        if self.ptr != UnsafePointer[T]():
            free_host(self.ptr.bitcast[UInt8]())
        self.ptr = ptr


# Equivalent to cms::cuda::host::unique_ptr<T>
alias unique_ptr[T] = OwnedPointer[_HostAllocation[T]]


# --- Primary factories (mirror C++ names exactly) ---
#
# NOTE: C++ distinguishes scalar vs array at the type level:
#   make_host_unique<T>(stream)    -> host::unique_ptr<T>
#   make_host_unique<T[]>(n,stream)-> host::unique_ptr<T[]>  (different type)
# The template specialization on T[] enforces this statically and gates
# operator* vs operator[] on the resulting pointer.
#
# Mojo has no T[] type parameter syntax, so both cases map to the same
# unique_ptr[T]. The scalar vs array distinction is carried only by which
# overload is called (presence of n: UInt), not by the type. A unique_ptr[T]
# that owns an array is indistinguishable from one that owns a single T.

fn make_host_unique[T](stream: cudaStream_t = cudaStreamDefault) -> unique_ptr[T] raises:
    let mem = allocate_host(UInt(sizeof[T]()), stream)
    return unique_ptr[T](_HostAllocation[T](mem.bitcast[T]()))


fn make_host_unique[T](
    n: UInt,
    stream: cudaStream_t = cudaStreamDefault,
) -> unique_ptr[T] raises:
    let mem = allocate_host(n * UInt(sizeof[T]()), stream)
    return unique_ptr[T](_HostAllocation[T](mem.bitcast[T]()))


fn make_host_unique_uninitialized[T](
    stream: cudaStream_t = cudaStreamDefault,
) -> unique_ptr[T] raises:
    let mem = allocate_host(UInt(sizeof[T]()), stream)
    return unique_ptr[T](_HostAllocation[T](mem.bitcast[T]()))


fn make_host_unique_uninitialized[T](
    n: UInt,
    stream: cudaStream_t = cudaStreamDefault,
) -> unique_ptr[T] raises:
    let mem = allocate_host(n * UInt(sizeof[T]()), stream)
    return unique_ptr[T](_HostAllocation[T](mem.bitcast[T]()))
