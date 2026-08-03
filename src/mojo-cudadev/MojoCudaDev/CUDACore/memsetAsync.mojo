# Mojo port of CUDACore/memsetAsync.h.
#
# cudaMemsetAsync has no direct Mojo binding that can target an arbitrary,
# caller-chosen stream: every overload of `DeviceContext.enqueue_memset`
# rejects a `stream=` keyword (confirmed empirically, same finding as
# copyAsync.mojo). So this is a tiny byte-fill kernel launched via the same
# compile_function + stream.enqueue_function idiom as launch.mojo, matching
# CUDA's own cudaMemsetAsync semantics of filling each *byte* with `value`
# (not each T-sized element with a T value -- the same footgun the C++ header
# comment warns about for sizeof(T) > 1 and value != 0).
#
# The array overload (`device::unique_ptr<T[]>&, int, size_t, stream`) is the
# one `cudadev` actually calls (`SiPixelDigiErrorsCUDA.cc:14`); the
# single-element overload (`device::unique_ptr<T>&, T value, stream`) below
# has no real call site in `cudadev` to verify against, so treat it as
# unverified-by-usage even though it compiles and follows the same idiom.
#
# C++'s single-element overload relies on an implicit conversion from `T
# value` to `cudaMemsetAsync`'s `int` parameter -- valid only for scalar-like
# T. Ported via Mojo's `Intable` trait (`Int(value)`), the direct equivalent
# of "T converts to int". C++ also guards this overload with
# `static_assert(std::is_array<T>::value == false, ...)`; no equivalent guard
# is needed here since a `T: Intable` generic can never be a raw array type in
# the first place.
from std.gpu.host import DeviceContext
from std.gpu.host.dim import Dim
from std.sys.info import size_of

from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType
from MojoCudaDev.CUDACore.currentDevice import currentDevice
from MojoCudaDev.CUDACore.device_unique_ptr import unique_ptr as DeviceUniquePtr


alias _threadsPerBlock = 256


fn _memset_kernel[T: AnyType](
    ptr: UnsafePointer[T, MutAnyOrigin],
    value: Int32,
    nbytes: Int32,
):
    from std.gpu import block_dim, block_idx, thread_idx

    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(nbytes):
        ptr.bitcast[UInt8]()[i] = UInt8(value & 0xFF)


fn memsetAsync[
    T: AnyType
](
    mut ptr: DeviceUniquePtr[T],
    value: Int32,
    nelements: UInt,
    stream: CUDAStreamType,
) raises:
    var nbytes = Int(nelements) * size_of[T]()
    var ctx = DeviceContext(api="cuda", device_id=currentDevice())
    var device_stream = stream.get()
    var compiled = ctx.compile_function[_memset_kernel[T], _memset_kernel[T]]()
    var blocks = (nbytes + _threadsPerBlock - 1) // _threadsPerBlock
    device_stream.enqueue_function(
        compiled,
        ptr.get(),
        value,
        Int32(nbytes),
        grid_dim=Dim(blocks),
        block_dim=Dim(_threadsPerBlock),
    )


# C++: memsetAsync(device::unique_ptr<T>& ptr, T value, cudaStream_t stream) -- memsetAsync.h:11-18
fn memsetAsync[
    T: Intable
](
    mut ptr: DeviceUniquePtr[T],
    value: T,
    stream: CUDAStreamType,
) raises:
    var nbytes = size_of[T]()
    var ctx = DeviceContext(api="cuda", device_id=currentDevice())
    var device_stream = stream.get()
    var compiled = ctx.compile_function[_memset_kernel[T], _memset_kernel[T]]()
    var blocks = (nbytes + _threadsPerBlock - 1) // _threadsPerBlock
    device_stream.enqueue_function(
        compiled,
        ptr.get(),
        Int32(Int(value)),
        Int32(nbytes),
        grid_dim=Dim(blocks),
        block_dim=Dim(_threadsPerBlock),
    )
