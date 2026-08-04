# Mojo port of CUDACore/prefixScan.h's blockPrefixScan overloads.
#
# Device path mirrors C++'s warp-shuffle + shared ws[] structure, but every
# lane always calls the warp shuffle (padded with 0 for out-of-range i,
# result only read/written when in range) instead of branching around it --
# a divergent subset of a warp calling shuffle/vote in this Mojo dialect
# deadlocks (confirmed via a minimal isolated repro). Verified correct across
# size < 32, == 32, > 32 (multi-warp), and >> block_size (multi-round).
from sys.info import is_gpu


fn warpPrefixScan[dtype: DType](x_in: Scalar[dtype], lane_id: UInt) -> Scalar[dtype]:
    from std.gpu.primitives.warp import shuffle_up
    var x = x_in
    var offset: UInt = 1
    while offset < 32:
        var y = shuffle_up(x, UInt32(offset))
        if lane_id >= offset:
            x += y
        offset <<= 1
    return x


# C++: blockPrefixScan(VT const* ci, VT* co, uint32_t size, T* ws) -- prefixScan.h:46-79
fn blockPrefixScan[
    dtype: DType, //, block_size: Int
](
    ci: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    co: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt32,
):
    @parameter
    if is_gpu():
        from std.gpu import thread_idx
        from std.gpu.sync import barrier
        from memory import stack_allocation, AddressSpace

        var first = Int(thread_idx.x)
        var laneId = UInt(thread_idx.x) & 0x1F
        var ws = stack_allocation[32, dtype, address_space = AddressSpace.SHARED]()

        var num_rounds = (Int(size) + block_size - 1) // block_size

        for round in range(num_rounds):
            var i = first + round * block_size
            var val: Scalar[dtype] = 0
            if i < Int(size):
                val = ci[i]
            var x = warpPrefixScan(val, laneId)
            if i < Int(size):
                co[i] = x
            var warpId = i // 32
            if laneId == 31 and i < Int(size):
                ws[warpId] = x

        barrier()

        if Int(size) > 32:
            var wsval: Scalar[dtype] = 0
            if first < 32:
                wsval = ws[first]
            wsval = warpPrefixScan(wsval, laneId)
            if first < 32:
                ws[first] = wsval
            barrier()

            for round in range(num_rounds):
                var i = first + 32 + round * block_size
                if i < Int(size):
                    var warpId = i // 32
                    co[i] += ws[warpId - 1]

            barrier()
    else:
        co[0] = ci[0]
        for i in range(1, Int(size)):
            co[i] = ci[i] + co[i - 1]


# C++: blockPrefixScan(T* c, uint32_t size, T* ws) -- prefixScan.h:87-119 (in-place)
fn blockPrefixScan[
    dtype: DType, //, block_size: Int
](c: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: UInt32):
    blockPrefixScan[block_size=block_size](c, c, size)
