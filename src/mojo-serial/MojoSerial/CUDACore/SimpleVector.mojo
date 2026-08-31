from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct SimpleVector[T: Movable & Copyable, DT: StaticString](
    Copyable, Defaultable, Movable, Sized, Typeable
):
    var m_size: Int32
    var m_capacity: Int32
    var m_data: UnsafePointer[Self.T]

    @always_inline
    def __init__(out self):
        self.m_size = 0
        self.m_capacity = 0
        self.m_data = UnsafePointer[T]()

    @always_inline
    def construct(mut self, var capacity: Int32, var data: UnsafePointer[Self.T]):
        # ownership of m_data stays within the caller
        self.m_size = 0
        self.m_capacity = capacity
        self.m_data = data

    @always_inline
    def push_back_unsafe(mut self, ref element: Self.T) -> Int32:
        var previousSize = self.m_size
        self.m_size += 1

        if previousSize < self.m_capacity:
            self.m_data[previousSize] = element
            return previousSize
        else:
            self.m_size -= 1
            return -1

    @always_inline
    def back(ref self) -> ref [self.m_data] Self.T:
        if self.m_size > 0:
            return self.m_data[self.m_size - 1]
        return self.m_data[]  # undefined behavior

    def push_back(mut self, ref element: Self.T) -> Int32:
        return self.push_back_unsafe(element)

    def extend(mut self, size: Int32 = 1) -> Int32:
        var previousSize = self.m_size
        self.m_size += size

        if previousSize < self.m_capacity:
            return previousSize
        else:
            self.m_size -= 1
            return -1

    def shrink(mut self, size: Int32 = 1) -> Int32:
        var previousSize = self.m_size
        self.m_size -= size

        if previousSize >= size:
            return previousSize - size
        else:
            self.m_size += size
            return -1

    @always_inline
    def empty(self) -> Bool:
        return self.m_size <= 0

    @always_inline
    def full(self) -> Bool:
        return self.m_size >= self.m_capacity

    @always_inline
    def __getitem__(ref self, i: Int32) -> ref [self.m_data] Self.T:
        return self.m_data[i]

    @always_inline
    def __setitem__(mut self, i: Int32, val: Self.T):
        self.m_data[i] = val

    @always_inline
    def reset(mut self):
        self.m_size = 0

    @always_inline
    def size(self) -> Int32:
        return self.m_size

    @always_inline
    def capacity(self) -> Int32:
        return self.m_capacity

    @always_inline
    def data(self) -> UnsafePointer[Self.T, mut=False]:
        return self.m_data

    @always_inline
    def resize(mut self, size: Int32):
        self.m_size = size

    @always_inline
    def set_data(mut self, data: UnsafePointer[Self.T]):
        self.m_data = data

    @always_inline
    def __len__(self) -> Int:
        return Int(self.m_size)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SimpleVector[" + Self.DT + "]"


def make_SimpleVector[
    T: Movable & Copyable, DT: StaticString
](var capacity: Int32, var data: UnsafePointer[T]) -> SimpleVector[T, DT]:
    var ret = SimpleVector[T, DT]()
    ret.construct(capacity, data)
    return ret


def make_SimpleVector[
    T: Movable & Copyable & Typeable, //
](var capacity: Int32, var data: UnsafePointer[T]) -> SimpleVector[
    T, T.dtype()
]:
    var ret = SimpleVector[T, T.dtype()]()
    ret.construct(capacity, data)
    return ret


def make_SimpleVector[
    T: Movable & Copyable, DT: StaticString, //
](
    mut mem: UnsafePointer[SimpleVector[T, DT]],
    var capacity: Int32,
    var data: UnsafePointer[T],
) -> ref [mem[]] SimpleVector[T, DT]:
    # construct a new object where mem points, assuming it is initialized
    mem.init_pointee_move(SimpleVector[T, DT]())
    mem[].construct(capacity, data)
    return mem[]
