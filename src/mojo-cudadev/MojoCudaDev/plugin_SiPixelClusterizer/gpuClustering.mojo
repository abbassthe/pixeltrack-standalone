from collections import InlineArray
from memory import stack_allocation
from std.memory import AddressSpace
from std.sys.info import is_gpu
from std.gpu.sync import barrier
from sys import is_defined

from MojoCudaDev.CUDACore.CUDAAtomics import (
    atomic_fetch_add,
    atomic_fetch_inc_wrap,
    atomic_fetch_min,
    atomic_fetch_min_block,
)
from MojoCudaDev.CUDACore.CUDASync import syncthreads_or, syncthreads_and
from MojoCudaDev.CUDACore.HistoContainer import HistoContainer
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering as gpuClusteringConstants
from MojoCudaDev.Geometry.Phase1PixelTopology import Phase1PixelTopology


# C++ declares these inside findClus's module loop (gpuClustering.h:78-80);
# Mojo has no function-local type alias, so they live here.
comptime maxPixInModule = 4000
comptime nbins = Int(Phase1PixelTopology.numColsInModule) + 2
comptime Hist = HistoContainer[DType.uint16, nbins, maxPixInModule, 9, UInt16]

# gpuClustering.h:136 -- the __CUDA_ARCH__ branch; this port is CUDA-only.
comptime maxiter = 16
comptime maxNeighbours = 10


struct gpuClustering:
    @staticmethod
    fn countModules(
        id: UnsafePointer[UInt16, MutAnyOrigin],
        moduleStart: UnsafePointer[UInt32, MutAnyOrigin],
        clusterId: UnsafePointer[Int32, MutAnyOrigin],
        numElements: Int32,
    ):
        from std.gpu import thread_idx, block_idx, block_dim, grid_dim

        var first = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        var stride = Int(grid_dim.x) * Int(block_dim.x)

        var i = first
        while i < Int(numElements):
            clusterId[i] = Int32(i)
            if gpuClusteringConstants.invalidModuleId != id[i]:
                var j = i - 1
                while j >= 0 and id[j] == gpuClusteringConstants.invalidModuleId:
                    j -= 1
                if j < 0 or id[j] != id[i]:
                    var loc = atomic_fetch_inc_wrap(moduleStart, UInt32(gpuClusteringConstants.maxNumModules))
                    moduleStart[Int(loc) + 1] = UInt32(i)
            i += stride

    # C++: findClus -- gpuClustering.h:38-309. gMaxHit is a parameter here;
    # C++ has it as a __device__ global (h:15), which this dialect rejects.
    @staticmethod
    fn findClus(
        id: UnsafePointer[UInt16, MutAnyOrigin],
        x: UnsafePointer[UInt16, MutAnyOrigin],
        y: UnsafePointer[UInt16, MutAnyOrigin],
        moduleStart: UnsafePointer[UInt32, MutAnyOrigin],
        nClustersInModule: UnsafePointer[UInt32, MutAnyOrigin],
        moduleId: UnsafePointer[UInt32, MutAnyOrigin],
        clusterId: UnsafePointer[Int32, MutAnyOrigin],
        numElements: Int32,
        gMaxHit: UnsafePointer[UInt32, MutAnyOrigin],
    ):
        from std.gpu import thread_idx, block_idx, block_dim, grid_dim

        # __shared__ declarations, hoisted out of the module loop (CUDA
        # allocates them once regardless). hist needs the generic-space cast:
        # a mut reference cannot be formed through a SHARED pointer here.
        var msize_sh = stack_allocation[
            1, Int32, address_space = AddressSpace.SHARED
        ]()
        var sh_hist = stack_allocation[
            1, Hist, address_space = AddressSpace.SHARED
        ]()
        var hist = sh_hist.address_space_cast[AddressSpace.GENERIC]()
        var ws = stack_allocation[
            32, Hist.Counter, address_space = AddressSpace.SHARED
        ]()
        var foundClusters = stack_allocation[
            1, UInt32, address_space = AddressSpace.SHARED
        ]()

        var firstModule = Int(block_idx.x)
        var endModule = Int(moduleStart[0])

        var module = firstModule
        while module < endModule:
            var firstPixel = Int(moduleStart[1 + module])
            var thisModuleId = id[firstPixel]
            debug_assert(
                Int(thisModuleId) < Int(gpuClusteringConstants.maxNumModules),
                "findClus: thisModuleId out of range",
            )

            @parameter
            if is_defined["GPU_DEBUG"]():
                if Int(thisModuleId) % 100 == 1:
                    if thread_idx.x == 0:
                        print(
                            "start clusterizer for module",
                            thisModuleId,
                            "in block",
                            block_idx.x,
                        )

            var first = firstPixel + Int(thread_idx.x)

            # find the index of the first pixel not belonging to this module
            msize_sh[0] = numElements
            barrier()

            # skip threads not associated to an existing pixel
            var i = first
            while i < Int(numElements):
                if id[i] != gpuClusteringConstants.invalidModuleId:
                    if id[i] != thisModuleId:
                        _ = atomic_fetch_min(msize_sh, Int32(i))
                        break
                i += Int(block_dim.x)

            # init hist (ymax=416 < 512 : 9bits)
            var j = Int(thread_idx.x)
            while j < Hist.totbins():
                hist[].base.off.data()[j] = 0
                j += Int(block_dim.x)
            barrier()

            debug_assert(
                msize_sh[0] == numElements
                or (
                    msize_sh[0] < numElements
                    and id[Int(msize_sh[0])] != thisModuleId
                ),
                "findClus: msize inconsistent",
            )

            # limit to maxPixInModule
            if thread_idx.x == 0:
                if Int(msize_sh[0]) - firstPixel > maxPixInModule:
                    print(
                        "too many pixels in module",
                        thisModuleId,
                        ":",
                        Int(msize_sh[0]) - firstPixel,
                        ">",
                        maxPixInModule,
                    )
                    msize_sh[0] = Int32(maxPixInModule + firstPixel)

            barrier()
            var msize = Int(msize_sh[0])
            debug_assert(
                msize - firstPixel <= maxPixInModule,
                "findClus: msize exceeds maxPixInModule",
            )


            # C++ declares totGood inside the #ifdef (h:102). A `@parameter if`
            # opens a scope but still name-resolves its body, so a declaration
            # in there would not reach the uses at h:113/h:123.
            var totGood = stack_allocation[
                1, UInt32, address_space = AddressSpace.SHARED
            ]()

            @parameter
            if is_defined["GPU_DEBUG"]():
                totGood[0] = 0
                barrier()

            # fill histo
            i = first
            while i < msize:
                if id[i] != gpuClusteringConstants.invalidModuleId:
                    hist[].count(y[i])

                    @parameter
                    if is_defined["GPU_DEBUG"]():
                        _ = atomic_fetch_add(totGood, UInt32(1))
                i += Int(block_dim.x)
            barrier()
            if thread_idx.x < 32:
                ws[Int(thread_idx.x)] = 0  # used by prefix scan...
            barrier()
            hist[].finalize(ws)
            barrier()

            @parameter
            if is_defined["GPU_DEBUG"]():
                debug_assert(
                    hist[].size() == totGood[0], "findClus: hist size mismatch"
                )
                if Int(thisModuleId) % 100 == 1:
                    if thread_idx.x == 0:
                        print("histo size", hist[].size())

            i = first
            while i < msize:
                if id[i] != gpuClusteringConstants.invalidModuleId:
                    hist[].fill(y[i], UInt16(i - firstPixel))
                i += Int(block_dim.x)

            # allocate space for duplicate pixels: a pixel can appear more than
            # once with different charge in the same event
            debug_assert(
                Int(hist[].size()) // Int(block_dim.x) <= maxiter,
                "findClus: hist.size()/blockDim.x exceeds maxiter",
            )
            # nearest neighbour -- per-thread, uninitialized like the C++
            var nn = InlineArray[
                InlineArray[UInt16, maxNeighbours], maxiter
            ](uninitialized=True)
            var nnn = InlineArray[UInt8, maxiter](uninitialized=True)
            for k in range(maxiter):
                nnn[k] = 0

            barrier()  # for hit filling!

            @parameter
            if is_defined["GPU_DEBUG"]():
                # look for anomalous high occupancy
                var n40 = stack_allocation[
                    1, UInt32, address_space = AddressSpace.SHARED
                ]()
                var n60 = stack_allocation[
                    1, UInt32, address_space = AddressSpace.SHARED
                ]()
                n40[0] = 0
                n60[0] = 0
                barrier()
                var jd = Int(thread_idx.x)
                while jd < Hist.nbins():
                    if hist[].size(jd) > 60:
                        _ = atomic_fetch_add(n60, UInt32(1))
                    if hist[].size(jd) > 40:
                        _ = atomic_fetch_add(n40, UInt32(1))
                    jd += Int(block_dim.x)
                barrier()
                if thread_idx.x == 0:
                    if n60[0] > 0:
                        print(
                            "columns with more than 60 px",
                            n60[0],
                            "in",
                            thisModuleId,
                        )
                    elif n40[0] > 0:
                        print(
                            "columns with more than 40 px",
                            n40[0],
                            "in",
                            thisModuleId,
                        )
                barrier()

            # fill NN
            var jn = Int(thread_idx.x)
            var kn = 0
            while jn < Int(hist[].size()):
                debug_assert(kn < maxiter, "findClus: k out of range")
                var p = hist[].begin() + jn
                var ii = Int(p[]) + firstPixel
                debug_assert(
                    id[ii] != gpuClusteringConstants.invalidModuleId,
                    "findClus: invalid pixel in hist",
                )
                debug_assert(
                    id[ii] == thisModuleId, "findClus: pixel in wrong module"
                )
                var be = Int(Hist.bin(y[ii] + 1))
                var e = hist[].end(be)
                p += 1
                debug_assert(nnn[kn] == 0, "findClus: nnn not zeroed")
                while p < e:
                    var m = Int(p[]) + firstPixel
                    debug_assert(m != ii, "findClus: m == i")
                    debug_assert(
                        Int(y[m]) - Int(y[ii]) >= 0, "findClus: y ordering"
                    )
                    debug_assert(
                        Int(y[m]) - Int(y[ii]) <= 1, "findClus: y gap > 1"
                    )
                    if abs(Int(x[m]) - Int(x[ii])) <= 1:
                        var l = Int(nnn[kn])
                        nnn[kn] += 1
                        debug_assert(
                            l < maxNeighbours, "findClus: too many neighbours"
                        )
                        nn[kn][l] = p[]
                    p += 1
                jn += Int(block_dim.x)
                kn += 1

            # for each pixel, look at all the pixels until the end of the
            # module; when two valid pixels within +/- 1 in x or y are found,
            # set their id to the minimum; after the loop, all the pixels in
            # each cluster should have the id equal to the lowest pixel in the
            # cluster (clus[i] == i).
            var more = True
            var nloops = 0
            while syncthreads_or(more):
                if 1 == nloops % 2:
                    var j2 = Int(thread_idx.x)
                    while j2 < Int(hist[].size()):
                        var p2 = hist[].begin() + j2
                        var i2 = Int(p2[]) + firstPixel
                        var m2 = clusterId[i2]
                        while m2 != clusterId[Int(m2)]:
                            m2 = clusterId[Int(m2)]
                        clusterId[i2] = m2
                        j2 += Int(block_dim.x)
                else:
                    more = False
                    var j3 = Int(thread_idx.x)
                    var k3 = 0
                    while j3 < Int(hist[].size()):
                        var p3 = hist[].begin() + j3
                        var i3 = Int(p3[]) + firstPixel
                        for kk in range(Int(nnn[k3])):
                            var l3 = nn[k3][kk]
                            var m3 = Int(l3) + firstPixel
                            debug_assert(m3 != i3, "findClus: m == i in nnloop")
                            #Assuming CUDA_ARCH >= 600 to match c++
                            var old = atomic_fetch_min_block(
                                clusterId + m3, clusterId[i3]
                            )
                            # do we need memory fence?
                            if old != clusterId[i3]:
                                # end the loop only if no changes were applied
                                more = True
                            _ = atomic_fetch_min_block(clusterId + i3, old)
                        j3 += Int(block_dim.x)
                        k3 += 1
                nloops += 1

            @parameter
            if is_defined["GPU_DEBUG"]():
                var n0 = stack_allocation[
                    1, Int32, address_space = AddressSpace.SHARED
                ]()
                if thread_idx.x == 0:
                    n0[0] = Int32(nloops)
                barrier()
                var ok = n0[0] == Int32(nloops)
                debug_assert(syncthreads_and(ok), "findClus: nloops diverged")
                if Int(thisModuleId) % 100 == 1:
                    if thread_idx.x == 0:
                        print("# loops", nloops)

            foundClusters[0] = 0
            barrier()

            # find the number of different clusters, identified by pixels with
            # clus[i] == i; mark these pixels with a negative id.
            i = first
            while i < msize:
                if id[i] != gpuClusteringConstants.invalidModuleId:
                    if clusterId[i] == Int32(i):
                        var old = atomic_fetch_inc_wrap(
                            foundClusters, UInt32(0xFFFFFFFF)
                        )
                        clusterId[i] = -(Int32(old) + 1)
                i += Int(block_dim.x)
            barrier()

            # propagate the negative id to all the pixels in the cluster.
            i = first
            while i < msize:
                if id[i] != gpuClusteringConstants.invalidModuleId:
                    if clusterId[i] >= 0:
                        # mark each pixel in a cluster with the same id as the
                        # first one
                        clusterId[i] = clusterId[Int(clusterId[i])]
                i += Int(block_dim.x)
            barrier()

            # adjust the cluster id to be a positive value starting from 0
            i = first
            while i < msize:
                if id[i] == gpuClusteringConstants.invalidModuleId:
                    clusterId[i] = -9999
                else:
                    clusterId[i] = -clusterId[i] - 1
                i += Int(block_dim.x)
            barrier()

            if thread_idx.x == 0:
                nClustersInModule[Int(thisModuleId)] = foundClusters[0]
                moduleId[module] = UInt32(thisModuleId)

                @parameter
                if is_defined["GPU_DEBUG"]():
                    if foundClusters[0] > gMaxHit[0]:
                        gMaxHit[0] = foundClusters[0]
                        if foundClusters[0] > 8:
                            print(
                                "max hit", foundClusters[0], "in", thisModuleId
                            )
                    if Int(thisModuleId) % 100 == 1:
                        print(
                            foundClusters[0],
                            "clusters in module",
                            thisModuleId,
                        )

            module += Int(grid_dim.x)