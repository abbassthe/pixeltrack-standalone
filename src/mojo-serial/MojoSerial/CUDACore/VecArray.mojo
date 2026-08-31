from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct VecArray[T: Movable & Copyable, DT: StaticString, maxSize: Int](
    Copyable, Defaultable, Movable, Sized, Typeable
):
    var m_data: InlineArray[Self.T, Self.maxSize, run_destructors=True]
    var m_size: Int32
    comptime ValueType = Self.T

    @always_inline
    def __init__(out self):
        self.m_data = InlineArray[T, maxSize, run_destructors=True](
            uninitialized=True
        )
        self.m_size = 0

    @always_inline
    def push_back_unsafe(mut self, ref element: Self.T) -> Int32:
        var previousSize = self.m_size
        self.m_size += 1

        if previousSize < Self.maxSize:
            self.m_data[previousSize] = element
            return previousSize
        else:
            self.m_size -= 1
            return -1

    @always_inline
    def back(self) -> ref [self.m_data] Self.T:
        if self.m_size > 0:
            return self.m_data[self.m_size - 1]
        else:
            return self.m_data[0]  # undefined behavior

    def push_back(mut self, ref element: Self.T) -> Int32:
        return self.push_back_unsafe(element)

    @always_inline
    def pop_back(mut self) -> Self.T:
        if self.m_size > 0:
            var previousSize = self.m_size
            self.m_size -= 1
            return self.m_data[previousSize - 1]
        else:
            return self.m_data[0]  # undefined behavior

    @always_inline
    def begin[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[Self.T, mut = origin.mut, origin=origin]:
        return self.m_data.unsafe_ptr()

    @always_inline
    def end[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[Self.T, mut = origin.mut, origin=origin]:
        return self.m_data.unsafe_ptr() + self.m_size

    @always_inline
    def __getitem__(ref self, i: Int32) -> ref [self.m_data] Self.T:
        return self.m_data[i]

    @always_inline
    def reset(mut self):
        self.m_size = 0

    @always_inline
    @staticmethod
    def capacity(self) -> Int32:
        return Self.maxSize

    @always_inline
    def data(self) -> UnsafePointer[Self.T, mut=False]:
        return self.m_data.unsafe_ptr()

    @always_inline
    def resize(mut self, var size: Int32):
        self.m_size = size

    @always_inline
    def empty(self) -> Bool:
        return self.m_size == 0

    @always_inline
    def full(self) -> Bool:
        return self.m_size == Self.maxSize

    @always_inline
    def __len__(self) -> Int:
        return Int(self.m_size)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "VecArray[" + Self.DT + ", " + Self.maxSize.__str__() + "]"
