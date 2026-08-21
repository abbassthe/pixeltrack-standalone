# Mojo port of CUDADataFormats/HeterogeneousSoA.h. Three mutually-exclusive
# owning pointers (dm_ptr/hm_ptr/std_ptr); get() returns whichever is set.
# std_ptr has no real caller in this codebase but is kept for fidelity to the
# C++ class. toHostAsync() takes an explicit allocator state param that C++
# doesn't need, matching make_host_unique/make_device_unique's own shape.
from MojoCudaDev.CUDACore.device_unique_ptr import unique_ptr as DeviceUniquePtr
from MojoCudaDev.CUDACore.host_unique_ptr import (
    unique_ptr as HostUniquePtr,
    make_host_unique,
    _HostAllocation,
)
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.MojoBridge.DTypes import Typeable

# copyAsync's own bound (AnyType) is looser than what this struct needs:
# __del__ must be able to destroy_pointee() on std_ptr, which needs real
# destructibility, not just AnyType. Typeable is needed by dtype() below.
comptime _HeterogeneousElement = Movable & ImplicitlyDestructible & Typeable


# OwnedPointer has no default constructor -- wrap a default (null) _HostAllocation instead.
fn _null_host_ptr[T: _HeterogeneousElement]() -> HostUniquePtr[T]:
    return HostUniquePtr[T](_HostAllocation[T]())


struct HeterogeneousSoA[T: _HeterogeneousElement](Defaultable, Movable, Typeable):
    comptime Product = Self.T

    var dm_ptr: DeviceUniquePtr[Self.T]
    var hm_ptr: HostUniquePtr[Self.T]
    var std_ptr: UnsafePointer[Self.T, MutAnyOrigin]

    fn __init__(out self):
        self.dm_ptr = DeviceUniquePtr[Self.T]()
        self.hm_ptr = _null_host_ptr[Self.T]()
        self.std_ptr = UnsafePointer[Self.T, MutAnyOrigin]()

    fn __init__(out self, var p: DeviceUniquePtr[Self.T]):
        self.dm_ptr = p^
        self.hm_ptr = _null_host_ptr[Self.T]()
        self.std_ptr = UnsafePointer[Self.T, MutAnyOrigin]()

    fn __init__(out self, var p: HostUniquePtr[Self.T]):
        self.dm_ptr = DeviceUniquePtr[Self.T]()
        self.hm_ptr = p^
        self.std_ptr = UnsafePointer[Self.T, MutAnyOrigin]()

    # Matches std::unique_ptr<T>&&: takes ownership, freed in __del__.
    fn __init__(out self, var p: UnsafePointer[Self.T, MutAnyOrigin]):
        self.dm_ptr = DeviceUniquePtr[Self.T]()
        self.hm_ptr = _null_host_ptr[Self.T]()
        self.std_ptr = p

    fn __moveinit__(out self, deinit take: Self):
        self.dm_ptr = take.dm_ptr^
        self.hm_ptr = take.hm_ptr^
        self.std_ptr = take.std_ptr
        take.std_ptr = UnsafePointer[Self.T, MutAnyOrigin]()

    fn __del__(deinit self):
        if self.std_ptr != UnsafePointer[Self.T, MutAnyOrigin]():
            self.std_ptr.destroy_pointee()
            self.std_ptr.free()

    fn get(self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        if self.dm_ptr.get() != UnsafePointer[Self.T, MutAnyOrigin]():
            return self.dm_ptr.get()
        if self.hm_ptr[].get() != UnsafePointer[Self.T, MutAnyOrigin]():
            return self.hm_ptr[].get()
        return self.std_ptr

    fn __getitem__(self) -> ref [self] Self.T:
        return self.get()[]

    # C++: cms::cuda::host::unique_ptr<T> toHostAsync(cudaStream_t) const --
    # HeterogeneousSoA.h:39-44
    fn toHostAsync(
        self, mut state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[Self.T]:
        debug_assert(
            self.dm_ptr.get() != UnsafePointer[Self.T, MutAnyOrigin](),
            "HeterogeneousSoA.toHostAsync: dm_ptr must be set",
        )
        var ret = make_host_unique[Self.T](state, stream)
        copyAsync[Self.T](ret, self.dm_ptr, stream)
        return ret^

    # ProductRegistry keys products by this string, so T must be encoded:
    # a constant name would collide across instantiations.
    @staticmethod
    fn dtype() -> String:
        return "HeterogeneousSoA[" + Self.T.dtype() + "]"
