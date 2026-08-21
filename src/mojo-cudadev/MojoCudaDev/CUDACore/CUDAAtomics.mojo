# Device atomics. See doc/MojoCudaDevAtomics.md before changing this file.

from os.atomic import Atomic, Consistency
from std.memory import AddressSpace
from sys.intrinsics import inlined_assembly


@always_inline
fn atomic_fetch_min_block(
    ptr: UnsafePointer[Int32, MutAnyOrigin], val: Int32
) -> Int32:
    """Atomically set `ptr[]` to min(ptr[], val) with block scope, returning the
    value held before.

    Mirrors CUDA `atomicMin_block(address, val)`. `os.atomic` exposes no memory
    scope, so this is the PTX instruction directly. Int32 only -- the mnemonic
    is type-specific and this is the only type used.
    """
    return inlined_assembly[
        "atom.cta.min.s32 $0, [$1], $2;",
        Int32,
        constraints="=r,l,r",
        has_side_effect=True,
    ](ptr, val)


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


@always_inline
fn atomic_fetch_inc_wrap[
    dt: DType, //, address_space: AddressSpace = AddressSpace.GENERIC
](ptr: UnsafePointer[Scalar[dt], MutAnyOrigin, address_space=address_space], bound: Scalar[dt]) -> Scalar[dt]:
    """Atomically set `ptr[]` to `0 if ptr[] >= bound else ptr[] + 1`, returning
    the value held before.

    Mirrors CUDA `atomicInc(address, bound)`. CAS retry loop, same reasoning
    as `atomic_fetch_min`.
    """
    var current = Atomic.load[ordering = Consistency.SEQUENTIAL](ptr)
    while True:
        var new_val = Scalar[dt](0) if current >= bound else current + 1
        var expected = current
        if Atomic.compare_exchange[
            success_ordering = Consistency.SEQUENTIAL,
            failure_ordering = Consistency.SEQUENTIAL,
        ](ptr, expected, new_val):
            return current
        current = expected
