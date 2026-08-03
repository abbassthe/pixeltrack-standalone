# Mojo port of CUDACore/host_noncached_unique_ptr.h.
#
# Unlike host_unique_ptr.mojo (which pools allocations through
# _AllocateHostState / the caching host allocator, meant for the per-event
# hot path), this allocates pinned host memory directly and frees it directly
# on destruction -- matching the C++ header's own doc comment: "these do not
# cache, so they should not be called per-event".
#
# The C++ side takes a CUDA host-alloc flags argument -- all 3 real call sites
# (BeamSpotToCUDA.cc, SiPixelRawToClusterGPUKernel.cu) pass
# cudaHostAllocWriteCombined, for staging buffers the host writes once and the
# device reads once. Mojo's high-level std.gpu.host API
# (DeviceContext.enqueue_create_host_buffer) has no flags parameter and can't
# honor this. Ported instead via raw FFI straight to the CUDA *driver* API
# (cuMemHostAlloc/cuMemFreeHost) -- libcuda.so is present on this system
# (libcudart.so, the runtime API library, is not installed at all), and the
# driver API's flag values are numerically identical to the runtime API's
# (cudaHostAllocWriteCombined == CU_MEMHOSTALLOC_WRITECOMBINED == 0x04), so the
# existing DTypes.mojo constants are reused directly without needing separate
# CU_* aliases.
#
# Verified end-to-end on real hardware: cuMemHostAlloc with the write-combined
# flag, a write, a read-back, and cuMemFreeHost, all returning CUDA_SUCCESS.
#
# This is the first file in this port to link against a CUDA library directly
# rather than going through std.gpu.host -- requires an explicit linker flag,
# since only the exact versioned filename libcuda.so.1 exists on this system
# (no unversioned libcuda.so dev-symlink): `-Xlinker -l:libcuda.so.1` in
# pixi.toml's run/build tasks.
from std.ffi import external_call
from std.sys.info import size_of

from MojoCudaDev.MojoBridge.DTypes import cudaHostAllocDefault


struct unique_ptr[T: AnyType](Movable):
    var _ptr: UnsafePointer[Self.T, MutAnyOrigin]

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Self.T, MutAnyOrigin]):
        self._ptr = ptr

    @always_inline
    fn __moveinit__(out self, var take: Self):
        self._ptr = take._ptr
        take._ptr = UnsafePointer[Self.T, MutAnyOrigin]()

    fn __del__(deinit self):
        if self._ptr != UnsafePointer[Self.T, MutAnyOrigin]():
            _ = external_call["cuMemFreeHost", Int32](self._ptr)

    @always_inline
    fn get(self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        return self._ptr


fn _cu_alloc[T: AnyType](nbytes: UInt, flags: UInt32) raises -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = UnsafePointer[T, MutAnyOrigin]()
    var pptr = UnsafePointer(to=ptr)
    var result = external_call["cuMemHostAlloc", Int32](pptr, nbytes, flags)
    if result != 0:
        raise Error("cuMemHostAlloc failed with CUresult " + String(result))
    return ptr


fn make_host_noncached_unique[
    T: AnyType
](flags: UInt32 = cudaHostAllocDefault) raises -> unique_ptr[T]:
    return unique_ptr[T](_cu_alloc[T](UInt(size_of[T]()), flags))


fn make_host_noncached_unique[
    T: AnyType
](n: UInt, flags: UInt32 = cudaHostAllocDefault) raises -> unique_ptr[T]:
    return unique_ptr[T](_cu_alloc[T](n * UInt(size_of[T]()), flags))
