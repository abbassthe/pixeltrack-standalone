from std.gpu.host import DeviceContext
from std.gpu.host.device_context import HostBuffer
from builtin.dtype import DType
from builtin.error import Error
from builtin.int import Int
from CUDACompat import CUDAStreamType, cudaStreamDefault
from CachingDeviceAllocator import CachingDeviceAllocator
from getCachingDeviceAllocator import binGrowth, maxBin
from utils.lock import BlockingSpinLock, BlockingScopedLock


alias HostPtr = UnsafePointer[UInt8]
alias ByteHostBuffer = HostBuffer[DType.uint8]
alias cudaStream_t = CUDAStreamType

var _allocated_host_buffers_lock = BlockingSpinLock()
var _allocated_host_buffers = List[Tuple[UInt, ByteHostBuffer]]()

fn _max_allocation_size() -> UInt:
    return CachingDeviceAllocator.IntPow(binGrowth, maxBin)


# Raw pinned-host allocation primitive. CachingHostAllocator must use this
# to avoid recursing through the public allocate_host()/free_host() wrapper.
fn allocate_host_raw(
    nbytes: UInt,
    stream: cudaStream_t = cudaStreamDefault,
    device: Int32 = Int32(-1)
) -> HostPtr:
    _ = stream
    if nbytes == 0:
        return HostPtr()

    try:
        var ctx = DeviceContext()
        if device >= Int32(0):
            ctx = DeviceContext(device_id = Int(device))
        var buf = ctx.enqueue_create_host_buffer[DType.uint8](Int(nbytes))
        ctx.synchronize()          # ensure allocation is complete

        var ptr = buf.unsafe_ptr()
        if ptr != HostPtr():
            with BlockingScopedLock(_allocated_host_buffers_lock):
                _allocated_host_buffers.append((UInt(ptr.address), buf))
        return ptr
    except e:
        return HostPtr()


fn free_host_raw(ptr: HostPtr):
    if ptr == HostPtr():
        return

    with BlockingScopedLock(_allocated_host_buffers_lock):
        var target = UInt(ptr.address)
        var i = 0
        while i < _allocated_host_buffers.__len__():
            if _allocated_host_buffers[i][0] == target:
                _allocated_host_buffers.remove(i)
                break
            i += 1


# Public API matching CUDACore/allocate_host.{h,cc}.
fn allocate_host(
    nbytes: UInt,
    stream: cudaStream_t = cudaStreamDefault
) -> HostPtr raises:
    let max_allocation_size = _max_allocation_size()
    if nbytes > max_allocation_size:
        raise Error(
            "Tried to allocate "
            + String(nbytes)
            + " bytes, but the allocator maximum is "
            + String(max_allocation_size)
        )
    return allocate_host_raw(nbytes, stream)


# Public API matching CUDACore/allocate_host.{h,cc}.
fn free_host(ptr: HostPtr):
    free_host_raw(ptr)
