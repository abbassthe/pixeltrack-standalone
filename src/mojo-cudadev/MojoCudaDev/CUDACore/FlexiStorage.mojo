# Mojo port of CUDACore/FlexiStorage.h. C++ picks fixed-array vs
# pointer+runtime-size storage via template specialization on S == -1; Mojo
# can't switch a struct's fields based on a parameter value, so both field
# sets are always present here and every method picks one via `@parameter if`.
fn _storage_size(s: Int) -> Int:
    if s < 0:
        return 1
    return s


struct FlexiStorage[
    I: Copyable & Movable & ImplicitlyCopyable & Defaultable, S: Int
](Movable):
    var _fixed: InlineArray[Self.I, _storage_size(Self.S)]
    var _ptr: UnsafePointer[Self.I, MutAnyOrigin]
    var _capacity: Int

    fn __init__(out self):
        self._fixed = InlineArray[Self.I, _storage_size(Self.S)](fill=Self.I())
        self._ptr = UnsafePointer[Self.I, MutAnyOrigin]()
        self._capacity = 0

    # C++: only exists on the S == -1 specialization; a no-op to call in
    # fixed-size mode since self._fixed is already valid storage.
    fn init(mut self, ptr: UnsafePointer[Self.I, MutAnyOrigin], capacity: Int):
        self._ptr = ptr
        self._capacity = capacity

    fn capacity(self) -> Int:
        @parameter
        if Self.S >= 0:
            return Self.S
        else:
            return self._capacity

    fn __getitem__(self, i: Int) -> Self.I:
        @parameter
        if Self.S >= 0:
            return self._fixed[i]
        else:
            return self._ptr[i]

    fn __setitem__(mut self, i: Int, value: Self.I):
        @parameter
        if Self.S >= 0:
            self._fixed[i] = value
        else:
            self._ptr[i] = value

    fn data(mut self) -> UnsafePointer[Self.I, MutAnyOrigin]:
        @parameter
        if Self.S >= 0:
            return self._fixed.unsafe_ptr()
        else:
            return self._ptr
