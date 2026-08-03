# Mojo port of CUDADataFormats/HostProduct.h.
#
# Heterogeneous host-side product: may hold a pinned host_unique_ptr[T] (from
# a GPU D2H transfer) or a plain heap OwnedPointer[T] (the C++ std::unique_ptr
# fallback, used by the CPU-only path). Both wrapped in Optional here because
# neither Mojo pointer type is itself nullable/absent by default the way
# C++'s std::unique_ptr / host::unique_ptr can be -- OwnedPointer always owns
# a concrete value once constructed, it has no empty state of its own.
from memory import OwnedPointer

from MojoCudaDev.CUDACore.host_unique_ptr import unique_ptr as HostUniquePtr


struct HostProduct[T: AnyType](Defaultable, Movable):
    var _hm_ptr: Optional[HostUniquePtr[Self.T]]
    var _std_ptr: Optional[OwnedPointer[Self.T]]

    @always_inline
    fn __init__(out self):
        self._hm_ptr = None
        self._std_ptr = None

    @always_inline
    fn __init__(out self, var p: HostUniquePtr[Self.T]):
        self._hm_ptr = p^
        self._std_ptr = None

    @always_inline
    fn __init__(out self, var p: OwnedPointer[Self.T]):
        self._hm_ptr = None
        self._std_ptr = p^

    @always_inline
    fn __moveinit__(out self, deinit take: Self):
        self._hm_ptr = take._hm_ptr^
        self._std_ptr = take._std_ptr^

    fn get(mut self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        if self._hm_ptr:
            return self._hm_ptr.value()[].get()
        if self._std_ptr:
            return self._std_ptr.value().unsafe_ptr()
        return UnsafePointer[Self.T, MutAnyOrigin]()
