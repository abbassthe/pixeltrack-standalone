from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add, atomic_fetch_sub
from MojoCudaDev.MojoBridge.DTypes import Typeable


@fieldwise_init
struct VecArray[T: Movable & Copyable, DT: StaticString, maxSize: Int](
    Copyable, Defaultable, Movable, Sized, Typeable
):
    var m_data: InlineArray[T, maxSize, run_destructors=True]
    var m_size: Int32
    alias ValueType = T

    @always_inline
    fn __init__(out self):
        self.m_data = InlineArray[T, maxSize, run_destructors=True](
            uninitialized=True
        )
        self.m_size = 0

    @always_inline
    fn push_back_unsafe(mut self, ref element: T) -> Int32:
        var previousSize = self.m_size
        self.m_size += 1

        if previousSize < maxSize:
            self.m_data[previousSize] = element
            return previousSize
        else:
            self.m_size -= 1
            return -1

    @always_inline
    fn back(self) -> ref [self.m_data] T:
        if self.m_size > 0:
            return self.m_data[self.m_size - 1]
        else:
            return self.m_data[0]  # undefined behavior

    fn push_back(mut self, ref element: T) -> Int32:
        # thread-safe version of the vector, when used in a CUDA kernel
        var previousSize = atomic_fetch_add(
            UnsafePointer(to=self.m_size), Int32(1)
        )

        if previousSize < maxSize:
            self.m_data[previousSize] = element
            return previousSize
        else:
            _ = atomic_fetch_sub(UnsafePointer(to=self.m_size), Int32(1))
            return -1

    @always_inline
    fn pop_back(mut self) -> T:
        if self.m_size > 0:
            var previousSize = self.m_size
            self.m_size -= 1
            return self.m_data[previousSize - 1]
        else:
            return self.m_data[0]  # undefined behavior

    @always_inline
    fn begin[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[T, mut = origin.mut, origin=origin]:
        return self.m_data.unsafe_ptr()

    @always_inline
    fn end[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[T, mut = origin.mut, origin=origin]:
        return self.m_data.unsafe_ptr() + self.m_size

    @always_inline
    fn __getitem__(ref self, i: Int32) -> ref [self.m_data] T:
        return self.m_data[i]

    @always_inline
    fn reset(mut self):
        self.m_size = 0

    @always_inline
    @staticmethod
    fn capacity(self) -> Int32:
        return maxSize

    @always_inline
    fn data(self) -> UnsafePointer[T, mut=False]:
        return self.m_data.unsafe_ptr()

    @always_inline
    fn resize(mut self, var size: Int32):
        self.m_size = size

    @always_inline
    fn empty(self) -> Bool:
        return self.m_size == 0

    @always_inline
    fn full(self) -> Bool:
        return self.m_size == maxSize

    @always_inline
    fn __len__(self) -> Int:
        return Int(self.m_size)

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "VecArray[" + DT + ", " + maxSize.__str__() + "]"
