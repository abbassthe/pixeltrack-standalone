# Mojo port of CUDACore/allocate_device.h
from gpu.host import DeviceContext
from gpu.host.device_context import DeviceBuffer
from builtin.dtype import DType
from utils.lock import BlockingSpinLock, BlockingScopedLock


# cudaStream_t is an opaque CUDA stream handle in C++.
alias cudaStream_t = OpaquePointer

# Device pointer equivalent to void* in C++.
alias DevicePtr = UnsafePointer[UInt8]
alias ByteDeviceBuffer = DeviceBuffer[DType.uint8]

var _allocated_buffers_lock = BlockingSpinLock()
var _allocated_buffers = List[Tuple[UInt, ByteDeviceBuffer]]()


# Allocate device memory.
fn allocate_device(device: Int32, nbytes: UInt, stream: cudaStream_t) -> DevicePtr:
    _ = stream
    if nbytes == 0:
        return DevicePtr()
    # keep the DeviceBuffer alive as long as you intend to use the pointer.
    var ctx = DeviceContext(device_id = Int(device))
    var buffer = ctx.create_buffer_sync[DType.uint8](Int(nbytes))
    var ptr = buffer.unsafe_ptr()
    if ptr != DevicePtr():
        with BlockingScopedLock(_allocated_buffers_lock):
            _allocated_buffers.append((UInt(ptr.address), buffer))
    return ptr


# Free device memory (to be called from unique_ptr).
fn free_device(device: Int32, ptr: DevicePtr, stream: cudaStream_t):
    _ = device
    _ = stream
    if ptr == DevicePtr():
        return
    #TODO replace this with a set to make faster 
    with BlockingScopedLock(_allocated_buffers_lock):
        var target = UInt(ptr.address)
        var i = 0
        while i < _allocated_buffers.__len__():
            if _allocated_buffers[i][0] == target:
                _allocated_buffers.remove(i)
                break
            i += 1
