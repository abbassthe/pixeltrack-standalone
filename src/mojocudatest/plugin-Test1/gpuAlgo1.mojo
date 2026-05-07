from std.sys.info import size_of
from std.gpu.host import DeviceContext

from MojoBridge.DTypes import Typeable
from CUDACompat import CUDAStreamType, cudaStreamDefault
from CUDAAppContext import CUDAAppContext
from device_unique_ptr import (
    unique_ptr as device_unique_ptr,
    make_device_unique,
    _DeviceAllocation,
)
from host_unique_ptr import unique_ptr as host_unique_ptr


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
    fn __moveinit__(out self, deinit take: Self):
        self.ptr = take.ptr^

    @always_inline
    fn get(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.ptr[].get()

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TypeableFloatBuffer"


# ── GPU kernels ──────────────────────────────────────────────────────────────

fn copy_kernel_float(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    numElements: Int32,
):
    from std.gpu import thread_idx, block_idx, block_dim
    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(numElements):
        dst[i] = src[i]


fn vectorAdd_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    numElements: Int32,
):
    from std.gpu import thread_idx, block_idx, block_dim
    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(numElements):
        c[i] = a[i] + b[i]


fn vectorProd_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    numElements: Int32,
):
    from std.gpu import thread_idx, block_idx, block_dim
    var row = Int(thread_idx.y + block_idx.y * block_dim.y)
    var col = Int(thread_idx.x + block_idx.x * block_dim.x)
    var n = Int(numElements)
    if row < n and col < n:
        c[row * n + col] = a[row] * b[col]


fn matrixMul_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    numElements: Int32,
):
    from std.gpu import thread_idx, block_idx, block_dim
    var row = Int(thread_idx.y + block_idx.y * block_dim.y)
    var col = Int(thread_idx.x + block_idx.x * block_dim.x)
    var n = Int(numElements)
    if row < n and col < n:
        var tmp: Float32 = 0
        for i in range(n):
            tmp += a[row * n + i] * b[i * n + col]
        c[row * n + col] = tmp


fn matrixMulVector_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    numElements: Int32,
):
    from std.gpu import thread_idx, block_idx, block_dim
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
    var d_a = make_device_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.device_state, stream
    )
    var d_b = make_device_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.device_state, stream
    )
    var d_c = make_device_unique[Float32](
        UInt(NUM_VALUES), cuda_ctx.device_state, stream
    )

    # Initialize d_a and d_b using temporary device buffers filled via map_to_host,
    # then copy to managed allocations with a GPU kernel (MAX 0.26.2 lacks DeviceStream.enqueue_copy).
    var init_ctx = DeviceContext()
    var tmp_a = init_ctx.create_buffer_sync[DType.float32](NUM_VALUES)
    var tmp_b = init_ctx.create_buffer_sync[DType.float32](NUM_VALUES)
    with tmp_a.map_to_host() as ha:
        for i in range(NUM_VALUES):
            ha[i] = Float32(i)
    with tmp_b.map_to_host() as hb:
        for i in range(NUM_VALUES):
            hb[i] = Float32(i * i)

    alias copyBlock = 32
    alias copyGrid = (NUM_VALUES + copyBlock - 1) // copyBlock
    init_ctx.enqueue_function[copy_kernel_float, copy_kernel_float](
        tmp_a.unsafe_ptr(), d_a[].get(), Int32(NUM_VALUES),
        grid_dim=(copyGrid,), block_dim=(copyBlock,),
    )
    init_ctx.enqueue_function[copy_kernel_float, copy_kernel_float](
        tmp_b.unsafe_ptr(), d_b[].get(), Int32(NUM_VALUES),
        grid_dim=(copyGrid,), block_dim=(copyBlock,),
    )
    init_ctx.synchronize()

    alias threadsPerBlock = 32
    alias blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) // threadsPerBlock

    var gpu_ctx = DeviceContext()
    gpu_ctx.enqueue_function[vectorAdd_kernel, vectorAdd_kernel](
        d_a[].get(), d_b[].get(), d_c[].get(), Int32(NUM_VALUES),
        grid_dim=(blocksPerGrid,), block_dim=(threadsPerBlock,),
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

    gpu_ctx.enqueue_function[vectorProd_kernel, vectorProd_kernel](
        d_a[].get(), d_b[].get(), d_ma[].get(), Int32(NUM_VALUES),
        grid_dim=(blocksPerGrid_x, blocksPerGrid_y),
        block_dim=(threadsPerBlock_x, threadsPerBlock_y),
    )
    gpu_ctx.enqueue_function[vectorProd_kernel, vectorProd_kernel](
        d_a[].get(), d_c[].get(), d_mb[].get(), Int32(NUM_VALUES),
        grid_dim=(blocksPerGrid_x, blocksPerGrid_y),
        block_dim=(threadsPerBlock_x, threadsPerBlock_y),
    )
    gpu_ctx.enqueue_function[matrixMul_kernel, matrixMul_kernel](
        d_ma[].get(), d_mb[].get(), d_mc[].get(), Int32(NUM_VALUES),
        grid_dim=(blocksPerGrid_x, blocksPerGrid_y),
        block_dim=(threadsPerBlock_x, threadsPerBlock_y),
    )

    gpu_ctx.enqueue_function[matrixMulVector_kernel, matrixMulVector_kernel](
        d_mc[].get(), d_b[].get(), d_c[].get(), Int32(NUM_VALUES),
        grid_dim=(blocksPerGrid,), block_dim=(threadsPerBlock,),
    )

    return d_a^
