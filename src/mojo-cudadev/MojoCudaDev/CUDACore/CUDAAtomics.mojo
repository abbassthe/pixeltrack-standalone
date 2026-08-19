# Device atomics. See doc/MojoCudaDevAtomics.md before changing this file.

from os.atomic import Atomic, Consistency
from std.memory import AddressSpace


@always_inline
fn atomic_fetch_add[
    dt: DType, //, address_space: AddressSpace = AddressSpace.GENERIC
](ptr: UnsafePointer[Scalar[dt], MutAnyOrigin, address_space=address_space], val: Scalar[dt]) -> Scalar[dt]:
    """Atomically add `val` to `ptr[]`, returning the value held before the add.

    Mirrors CUDA `atomicAdd(address, val)`.
    """
    return Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](ptr, val)


@always_inline
fn atomic_fetch_sub[
    dt: DType, //, address_space: AddressSpace = AddressSpace.GENERIC
](ptr: UnsafePointer[Scalar[dt], MutAnyOrigin, address_space=address_space], val: Scalar[dt]) -> Scalar[dt]:
    """Atomically subtract `val` from `ptr[]`, returning the value held before.

    Mirrors CUDA `atomicSub(address, val)`.
    """
    return Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](ptr, -val)


@always_inline
fn atomic_fetch_min[
    dt: DType, //, address_space: AddressSpace = AddressSpace.GENERIC
](ptr: UnsafePointer[Scalar[dt], MutAnyOrigin, address_space=address_space], val: Scalar[dt]) -> Scalar[dt]:
    """Atomically set `ptr[]` to min(ptr[], val), returning the value held before.

    Mirrors CUDA `atomicMin(address, val)`. No fetch_min primitive exists in
    this Mojo version (only a non-returning `Atomic.min`), so this is a CAS
    retry loop on `Atomic.compare_exchange`.
    """
    var current = Atomic.load[ordering = Consistency.SEQUENTIAL](ptr)
    while current > val:
        var expected = current
        if Atomic.compare_exchange[
            success_ordering = Consistency.SEQUENTIAL,
            failure_ordering = Consistency.SEQUENTIAL,
        ](ptr, expected, val):
            return current
        current = expected
    return current
