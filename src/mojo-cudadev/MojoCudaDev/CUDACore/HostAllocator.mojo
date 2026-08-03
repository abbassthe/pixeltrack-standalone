# Mojo port of CUDACore/HostAllocator.h. Mojo's List has no pluggable-allocator
# mechanism (unlike std::vector<T, HostAllocator<T>>), so this is a fixed-size
# pinned buffer instead -- it does not support push_back/resize.
from MojoCudaDev.MojoBridge.DTypes import cudaHostAllocDefault
from MojoCudaDev.CUDACore.host_noncached_unique_ptr import (
    unique_ptr as HostNoncachedUniquePtr,
    make_host_noncached_unique,
)


struct HostAllocator[T: AnyType & Copyable & ImplicitlyCopyable](Movable):
    var _ptr: HostNoncachedUniquePtr[Self.T]
    var _size: Int

    fn __init__(out self, n: Int, flags: UInt32 = cudaHostAllocDefault) raises:
        self._ptr = make_host_noncached_unique[Self.T](UInt(n), flags)
        self._size = n

    fn __moveinit__(out self, deinit take: Self):
        self._ptr = take._ptr^
        self._size = take._size

    fn size(self) -> Int:
        return self._size

    fn data(self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        return self._ptr.get()

    fn __getitem__(self, i: Int) -> Self.T:
        return self.data()[i]

    fn __setitem__(mut self, i: Int, value: Self.T):
        self.data()[i] = value
