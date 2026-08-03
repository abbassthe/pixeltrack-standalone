# Mojo port of CUDACore/copyAsync.h.
#
# cudaMemcpyAsync has no direct Mojo binding that can target an arbitrary,
# caller-chosen stream: DeviceContext.enqueue_copy and
# DeviceBuffer/HostBuffer.enqueue_copy_to/enqueue_copy_from all only operate
# on the context's own implicit stream (confirmed empirically -- every
# overload of every copy-capable type in std.gpu.host rejects a `stream=`
# keyword; there is no lower-level API elsewhere either). So this is a tiny
# elementwise copy kernel launched via the same compile_function +
# stream.enqueue_function idiom as launch.mojo, which -- unlike copy/memset --
# does support an explicit stream. Host and device pointers are both plain
# UnsafePointer[T, MutAnyOrigin] in this Mojo GPU API (no distinct address
# space marker), so the same kernel body works for both directions; verified
# to compile with one pointer from enqueue_create_buffer and one from
# enqueue_create_host_buffer passed to the same kernel launch.
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

from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType
from MojoCudaDev.CUDACore.currentDevice import currentDevice
from MojoCudaDev.CUDACore.device_unique_ptr import unique_ptr as DeviceUniquePtr
from MojoCudaDev.CUDACore.host_unique_ptr import unique_ptr as HostUniquePtr


alias _threadsPerBlock = 256
alias _CopyElement = ImplicitlyCopyable & Movable & ImplicitlyDestructible


fn _copy_kernel[T: _CopyElement](
    dst: UnsafePointer[T, MutAnyOrigin],
    src: UnsafePointer[T, MutAnyOrigin],
    n: Int32,
):
    from std.gpu import block_dim, block_idx, thread_idx

    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(n):
        dst[i] = src[i]


fn _launch_copy[T: _CopyElement](
    dst: UnsafePointer[T, MutAnyOrigin],
    src: UnsafePointer[T, MutAnyOrigin],
    nelements: Int,
    stream: CUDAStreamType,
) raises:
    var ctx = DeviceContext(api="cuda", device_id=currentDevice())
    var device_stream = stream.get()
    var compiled = ctx.compile_function[_copy_kernel[T], _copy_kernel[T]]()
    var blocks = (nelements + _threadsPerBlock - 1) // _threadsPerBlock
    device_stream.enqueue_function(
        compiled,
        dst,
        src,
        Int32(nelements),
        grid_dim=Dim(blocks),
        block_dim=Dim(_threadsPerBlock),
    )


# Single element, host -> device.
fn copyAsync[T: _CopyElement](
    mut dst: DeviceUniquePtr[T], src: HostUniquePtr[T], stream: CUDAStreamType
) raises:
    _launch_copy[T](dst.get(), src[].get(), 1, stream)


# Single element, device -> host.
fn copyAsync[T: _CopyElement](
    mut dst: HostUniquePtr[T], src: DeviceUniquePtr[T], stream: CUDAStreamType
) raises:
    _launch_copy[T](dst[].get(), src.get(), 1, stream)


# Multiple elements, host -> device.
fn copyAsync[T: _CopyElement](
    mut dst: DeviceUniquePtr[T],
    src: HostUniquePtr[T],
    nelements: UInt,
    stream: CUDAStreamType,
) raises:
    _launch_copy[T](dst.get(), src[].get(), Int(nelements), stream)


# Multiple elements, device -> host.
fn copyAsync[T: _CopyElement](
    mut dst: HostUniquePtr[T],
    src: DeviceUniquePtr[T],
    nelements: UInt,
    stream: CUDAStreamType,
) raises:
    _launch_copy[T](dst[].get(), src.get(), Int(nelements), stream)
