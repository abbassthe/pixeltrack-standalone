# Device atomics. See doc/MojoCudaDevAtomics.md before changing this file.

from os.atomic import Atomic, Consistency


@always_inline
fn atomic_fetch_add[
    dt: DType, //
](ptr: UnsafePointer[Scalar[dt], mut=True], val: Scalar[dt]) -> Scalar[dt]:
    """Atomically add `val` to `ptr[]`, returning the value held before the add.

    Mirrors CUDA `atomicAdd(address, val)`.
    """
    return Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](ptr, val)


@always_inline
fn atomic_fetch_sub[
    dt: DType, //
](ptr: UnsafePointer[Scalar[dt], mut=True], val: Scalar[dt]) -> Scalar[dt]:
    """Atomically subtract `val` from `ptr[]`, returning the value held before.

    Mirrors CUDA `atomicSub(address, val)`.
    """
    return Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](ptr, -val)
