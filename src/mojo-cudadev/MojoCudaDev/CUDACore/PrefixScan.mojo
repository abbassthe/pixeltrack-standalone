# Mojo port of CUDACore/prefixScan.h's blockPrefixScan overloads.
#
# C++'s loop condition (`i < size`, checked per-thread) gives lanes within the
# same warp different trip counts once `size` isn't a multiple of block_size --
# a divergent subset of a warp then calling shuffle/vote deadlocks in this Mojo
# dialect (confirmed via a minimal isolated repro). Fixed by checking each
# warp's *base* index instead of each thread's own -- uniform across all 32
# lanes of a warp, so a warp's lanes always enter/exit its loop together, while
# different warps can still stop after different numbers of rounds. C++'s
# `mask` becomes `bound`: the in-range check moves inside warpPrefixScan
# itself, since this per-warp-uniform condition (not masking) is what makes
# every shuffle call safe. Verified correct including size < 32, == 32, > 32
# (multi-warp), >> block_size (multi-round), and many-empty-warps cases.
from sys.info import is_gpu
from std.memory import AddressSpace


# C++: warpPrefixScan(T const* ci, T* co, uint32_t i, uint32_t mask) -- prefixScan.h:14-24
fn warpPrefixScan[
    dtype: DType, address_space: AddressSpace
](
    ci: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    co: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    i: Int,
    bound: Int,
    laneId: UInt,
):
    from std.gpu.primitives.warp import shuffle_up

    var x: Scalar[dtype] = 0
    if i < bound:
        x = ci[i]

    var offset: UInt = 1
    while offset < 32:
        var y = shuffle_up(x, UInt32(offset))
        if laneId >= offset:
            x += y
        offset <<= 1

    if i < bound:
        co[i] = x


# C++: warpPrefixScan(T* c, uint32_t i, uint32_t mask) -- prefixScan.h:29-38 (in-place)
fn warpPrefixScan[
    dtype: DType, address_space: AddressSpace
](
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    i: Int,
    bound: Int,
    laneId: UInt,
):
    warpPrefixScan(c, c, i, bound, laneId)


# C++: blockPrefixScan(VT const* ci, VT* co, uint32_t size, T* ws) -- prefixScan.h:46-79
fn blockPrefixScan[
    dtype: DType, ws_address_space: AddressSpace, //
](
    ci: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    co: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt32,
    ws: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=ws_address_space],
):
    @parameter
    if is_gpu():
        from std.gpu import thread_idx, block_dim
        from std.gpu.sync import barrier

        var block_size = Int(block_dim.x)
        debug_assert(block_size % 32 == 0, "blockPrefixScan: blockDim.x must be a multiple of 32")
        debug_assert(
            ws != UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=ws_address_space](),
            "blockPrefixScan: ws must not be null",
        )
        debug_assert(size <= 1024, "blockPrefixScan: size must be <= 1024")

        var first = Int(thread_idx.x)
        var laneId = UInt(thread_idx.x) & 0x1F
        var warpBase = first - Int(laneId)

        var i = first
        while warpBase < Int(size):
            warpPrefixScan(ci, co, i, Int(size), laneId)
            var warpId = i // 32
            if laneId == 31 and i < Int(size):
                debug_assert(warpId < 32, "blockPrefixScan: warpId out of ws[] bounds")
                ws[warpId] = co[i]
            i += block_size
            warpBase += block_size

        barrier()

        if Int(size) > 32:
            warpPrefixScan(ws, first, 32, laneId)
            barrier()

            var i2 = first + 32
            var warpBase2 = i2 - Int(laneId)
            while warpBase2 < Int(size):
                var warpId = i2 // 32
                if i2 < Int(size):
                    co[i2] += ws[warpId - 1]
                i2 += block_size
                warpBase2 += block_size

            barrier()
    else:
        co[0] = ci[0]
        for i in range(1, Int(size)):
            co[i] = ci[i] + co[i - 1]


# C++: blockPrefixScan(T* c, uint32_t size, T* ws) -- prefixScan.h:87-119 (in-place)
fn blockPrefixScan[
    dtype: DType, ws_address_space: AddressSpace, //
](
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt32,
    ws: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=ws_address_space],
):
    blockPrefixScan(c, c, size, ws)
