# Mojo port of CUDACore/copyAsync.h.
#
# cudaMemcpyAsync has no direct Mojo binding that can target an arbitrary,
# caller-chosen stream: DeviceContext.enqueue_copy and
# DeviceBuffer/HostBuffer.enqueue_copy_to/enqueue_copy_from all only operate
# on the context's own implicit stream (confirmed empirically -- every
# overload of every copy-capable type in std.gpu.host rejects a `stream=`
# keyword; there is no lower-level API elsewhere either). So this is a tiny
# byte-copy kernel launched via the same compile_function +
# stream.enqueue_function idiom as launch.mojo, which -- unlike copy/memset --
# does support an explicit stream. Host and device pointers are both plain
# UnsafePointer[T, MutAnyOrigin] in this Mojo GPU API (no distinct address
# space marker), so the same kernel body works for both directions; verified
# to compile with one pointer from enqueue_create_buffer and one from
# enqueue_create_host_buffer passed to the same kernel launch.
#
# The kernel copies raw bytes (one thread per byte), matching C++'s
# `cudaMemcpyAsync(dst, src, sizeof(T)*n, ...)` -- not a per-element
# `dst[i] = src[i]`, which would require T: ImplicitlyCopyable and silently
# break for any T that owns non-copyable fields (e.g. OneToManyAssoc, used
# by TrackSoAHeterogeneousT via HitContainer). _CopyElement therefore only
# needs T: AnyType, matching what device_unique_ptr/host_unique_ptr require.
#
# Only the four overloads using device::unique_ptr / host::unique_ptr are
# ported (single- and multi-element, both directions). Not ported, since none
# of these types exist in this tree yet:
#   - host::noncached::unique_ptr overloads -- host_noncached_unique_ptr.mojo
#     is not written (Phase 1, still pending).
#   - the std::vector<T, HostAllocator<T>> overload -- HostAllocator.mojo is
#     not written (Phase 1, still pending).
#   - the propagate_const_array overload -- Framework/propagate_const_array.mojo
#     is not written (C5, still pending).
from std.gpu.host import DeviceContext
from std.gpu.host.dim import Dim
from std.sys.info import size_of

from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType
from MojoCudaDev.CUDACore.currentDevice import currentDevice
from MojoCudaDev.CUDACore.device_unique_ptr import unique_ptr as DeviceUniquePtr
from MojoCudaDev.CUDACore.host_unique_ptr import unique_ptr as HostUniquePtr


alias _threadsPerBlock = 256
alias _CopyElement = AnyType


fn _copy_kernel[T: _CopyElement](
    dst: UnsafePointer[T, MutAnyOrigin],
    src: UnsafePointer[T, MutAnyOrigin],
    total_bytes: Int64,
):
    from std.gpu import block_dim, block_idx, thread_idx

    var dst_bytes = dst.bitcast[UInt8]()
    var src_bytes = src.bitcast[UInt8]()
    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(total_bytes):
        dst_bytes[i] = src_bytes[i]


fn _launch_copy[T: _CopyElement](
    dst: UnsafePointer[T, MutAnyOrigin],
    src: UnsafePointer[T, MutAnyOrigin],
    nelements: Int,
    stream: CUDAStreamType,
) raises:
    var ctx = DeviceContext(api="cuda", device_id=currentDevice())
    var device_stream = stream.get()
    var compiled = ctx.compile_function[_copy_kernel[T], _copy_kernel[T]]()
    var total_bytes = nelements * size_of[T]()
    var blocks = (total_bytes + _threadsPerBlock - 1) // _threadsPerBlock
    device_stream.enqueue_function(
        compiled,
        dst,
        src,
        Int64(total_bytes),
        grid_dim=Dim(blocks),
        block_dim=Dim(_threadsPerBlock),
    )


# Single element, host -> device. src is borrowed, matching C++'s
# pass-by-reference -- use this when the caller keeps using its host source
# afterward (e.g. SiPixelDigiErrorsCUDA.mojo's error_h, a persistent field:
# nothing frees it early, so no synchronize is needed here). For an ephemeral
# host buffer that gets dropped right after this call, use copyAsyncOwned
# below instead: this port's host allocator frees synchronously and
# immediately (no stream-ordered deferral like C++'s real allocator), so a
# borrowed src that's freed right after this call could still be in flight.
fn copyAsync[T: _CopyElement](
    mut dst: DeviceUniquePtr[T], src: HostUniquePtr[T], stream: CUDAStreamType
) raises:
    _launch_copy[T](dst.get(), src[].get(), 1, stream)


# Single element, host -> device, owning src. Synchronizes internally before
# returning, then src is destroyed automatically -- safe by construction for
# an ephemeral staging buffer (see SiPixelDigisCUDA.mojo/
# SiPixelClustersCUDA.mojo's DeviceConstView staging), with no caller-side
# synchronize or lifetime-extension trick needed.
fn copyAsyncOwned[T: _CopyElement](
    mut dst: DeviceUniquePtr[T], var src: HostUniquePtr[T], stream: CUDAStreamType
) raises:
    _launch_copy[T](dst.get(), src[].get(), 1, stream)
    stream.synchronize()


# Single element, device -> host.
fn copyAsync[T: _CopyElement](
    mut dst: HostUniquePtr[T], src: DeviceUniquePtr[T], stream: CUDAStreamType
) raises:
    _launch_copy[T](dst[].get(), src.get(), 1, stream)


# Multiple elements, host -> device. See the single-element overload above
# for the borrowed-vs-owned distinction.
fn copyAsync[T: _CopyElement](
    mut dst: DeviceUniquePtr[T],
    src: HostUniquePtr[T],
    nelements: UInt,
    stream: CUDAStreamType,
) raises:
    _launch_copy[T](dst.get(), src[].get(), Int(nelements), stream)


# Multiple elements, host -> device, owning src. See copyAsyncOwned above.
fn copyAsyncOwned[T: _CopyElement](
    mut dst: DeviceUniquePtr[T],
    var src: HostUniquePtr[T],
    nelements: UInt,
    stream: CUDAStreamType,
) raises:
    _launch_copy[T](dst.get(), src[].get(), Int(nelements), stream)
    stream.synchronize()


# Multiple elements, device -> host.
fn copyAsync[T: _CopyElement](
    mut dst: HostUniquePtr[T],
    src: DeviceUniquePtr[T],
    nelements: UInt,
    stream: CUDAStreamType,
) raises:
    _launch_copy[T](dst[].get(), src.get(), Int(nelements), stream)
