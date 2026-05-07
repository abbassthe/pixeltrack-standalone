from sys import sizeof
from std.gpu.host import DeviceContext

from MojoBridge.DTypes import Typeable
from CUDACORE.cms.CUDA.CUDACompat import CUDAStreamType, cudaStreamDefault
from CUDACORE.cms.CUDA.CUDAAppContext import CUDAAppContext
from CUDACORE.cms.CUDA.device_unique_ptr import (
    unique_ptr as device_unique_ptr,
    make_device_unique,
    _DeviceAllocation,
)
from CUDACORE.cms.CUDA.host_unique_ptr import (
    unique_ptr as host_unique_ptr,
    make_host_unique,
)


alias cudaStream_t = CUDAStreamType
alias NUM_VALUES = 1000


# Typeable wrapper around device_unique_ptr[Float32] so it can be stored in
# a Product[T] (Product requires T: Movable & Typeable).
struct TypeableFloatBuffer(Defaultable, Movable, Typeable):
    var ptr: device_unique_ptr[Float32]

    @always_inline
    fn __init__(out self):
        self.ptr = device_unique_ptr[Float32](_DeviceAllocation[Float32]())

    @always_inline
    fn __init__(out self, var ptr: device_unique_ptr[Float32]):
        self.ptr = ptr^

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self.ptr = other.ptr^

    @always_inline
    fn get(self) -> UnsafePointer[Float32]:
        return self.ptr[].get()

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TypeableFloatBuffer"


# ── GPU kernels ──────────────────────────────────────────────────────────────

fn vectorAdd_kernel(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    numElements: Int32,
):
    from gpu import thread_idx, block_idx, block_dim
    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(numElements):
        c[i] = a[i] + b[i]


fn vectorProd_kernel(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    numElements: Int32,
):
    from gpu import thread_idx, block_idx, block_dim
    var row = Int(thread_idx.y + block_idx.y * block_dim.y)
    var col = Int(thread_idx.x + block_idx.x * block_dim.x)
    var n = Int(numElements)
    if row < n and col < n:
        c[row * n + col] = a[row] * b[col]


fn matrixMul_kernel(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    numElements: Int32,
):
    from gpu import thread_idx, block_idx, block_dim
    var row = Int(thread_idx.y + block_idx.y * block_dim.y)
    var col = Int(thread_idx.x + block_idx.x * block_dim.x)
    var n = Int(numElements)
    if row < n and col < n:
        var tmp: Float32 = 0
        for i in range(n):
            tmp += a[row * n + i] * b[i * n + col]
        c[row * n + col] = tmp


fn matrixMulVector_kernel(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    numElements: Int32,
):
    from gpu import thread_idx, block_idx, block_dim
    var row = Int(thread_idx.x + block_idx.x * block_dim.x)
    var n = Int(numElements)
    if row < n:
        var tmp: Float32 = 0
        for i in range(n):
            tmp += a[row * n + i] * b[i]
        c[row] = tmp


# ── Host-side algorithm ─────────────────────────────────────────────────────
fn gpuAlgo1(
    stream: cudaStream_t, mut cuda_ctx: CUDAAppContext
) raises -> device_unique_ptr[Float32]:
    var h_a = make_host_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.host_state, stream
    )
    var h_b = make_host_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.host_state, stream
    )

    for i in range(NUM_VALUES):
        h_a[].ptr[i] = Float32(i)
        h_b[].ptr[i] = Float32(i * i)

    var d_a = make_device_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.device_state, stream
    )
    var d_b = make_device_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.device_state, stream
    )
    var d_c = make_device_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.device_state, stream
    )

    var actual_stream = stream.get()
    actual_stream.enqueue_copy(
        d_a[].get().bitcast[UInt8](),
        h_a[].get().bitcast[UInt8](),
        NUM_VALUES * sizeof[Float32](),
    )
    actual_stream.enqueue_copy(
        d_b[].get().bitcast[UInt8](),
        h_b[].get().bitcast[UInt8](),
        NUM_VALUES * sizeof[Float32](),
    )

    alias threadsPerBlock = 32
    alias blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) // threadsPerBlock

    var gpu_ctx = DeviceContext()
    gpu_ctx.enqueue_function[vectorAdd_kernel](
        (blocksPerGrid,),
        (threadsPerBlock,),
        d_a[].get(),
        d_b[].get(),
        d_c[].get(),
        Int32(NUM_VALUES),
    )

    var d_ma = make_device_unique[Float32](
        UInt(NUM_VALUES * NUM_VALUES), cuda_ctx.device_state, stream
    )
    var d_mb = make_device_unique[Float32](
        UInt(NUM_VALUES * NUM_VALUES), cuda_ctx.device_state, stream
    )
    var d_mc = make_device_unique[Float32](
        UInt(NUM_VALUES * NUM_VALUES), cuda_ctx.device_state, stream
    )

    var threadsPerBlock_x: Int = NUM_VALUES
    var threadsPerBlock_y: Int = NUM_VALUES
    var blocksPerGrid_x: Int = 1
    var blocksPerGrid_y: Int = 1
    if NUM_VALUES * NUM_VALUES > 32:
        threadsPerBlock_x = 32
        threadsPerBlock_y = 32
        blocksPerGrid_x = (NUM_VALUES + 31) // 32
        blocksPerGrid_y = (NUM_VALUES + 31) // 32

    gpu_ctx.enqueue_function[vectorProd_kernel](
        (blocksPerGrid_x, blocksPerGrid_y),
        (threadsPerBlock_x, threadsPerBlock_y),
        d_a[].get(),
        d_b[].get(),
        d_ma[].get(),
        Int32(NUM_VALUES),
    )
    gpu_ctx.enqueue_function[vectorProd_kernel](
        (blocksPerGrid_x, blocksPerGrid_y),
        (threadsPerBlock_x, threadsPerBlock_y),
        d_a[].get(),
        d_c[].get(),
        d_mb[].get(),
        Int32(NUM_VALUES),
    )
    gpu_ctx.enqueue_function[matrixMul_kernel](
        (blocksPerGrid_x, blocksPerGrid_y),
        (threadsPerBlock_x, threadsPerBlock_y),
        d_ma[].get(),
        d_mb[].get(),
        d_mc[].get(),
        Int32(NUM_VALUES),
    )

    gpu_ctx.enqueue_function[matrixMulVector_kernel](
        (blocksPerGrid,),
        (threadsPerBlock,),
        d_mc[].get(),
        d_b[].get(),
        d_c[].get(),
        Int32(NUM_VALUES),
    )

    return d_a^
