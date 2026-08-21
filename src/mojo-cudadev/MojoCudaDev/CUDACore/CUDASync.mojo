# Device block-sync primitives. See doc/MojoCudaDevAtomics.md before changing this file.

from memory import stack_allocation
from std.memory import AddressSpace
from std.gpu.sync import barrier

from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add


@always_inline
fn syncthreads_or(more: Bool) -> Bool:
    """Block-wide barrier + OR-reduce: true if any thread in the block passed
    true. Mirrors CUDA `__syncthreads_or(predicate)`.
    """
    from std.gpu import thread_idx

    var flag = stack_allocation[1, Int32, address_space = AddressSpace.SHARED]()
    if thread_idx.x == 0:
        flag[0] = 0
    barrier()
    if more:
        _ = atomic_fetch_add(flag, Int32(1))
    barrier()
    var result = flag[0] != 0
    barrier()
    return result
