# GPU kernels from SiPixelRawToClusterGPUKernel.cu that are independent of
# raw-word decoding. RawToDigi will join this module when it is ported.
from memory import stack_allocation
from std.memory import AddressSpace
from std.gpu.sync import barrier

from MojoCudaDev.CUDACore.PrefixScan import blockPrefixScan
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering


struct SiPixelRawToClusterKernels:
    # C++: fillHitsModuleStart(). This must launch as exactly one 1024-thread
    # block: PrefixScan supports at most 1024 values per invocation.
    @staticmethod
    fn fillHitsModuleStart(
        cluStart: UnsafePointer[UInt32, MutAnyOrigin],
        moduleStart: UnsafePointer[UInt32, MutAnyOrigin],
    ):
        from std.gpu import block_dim, block_idx, grid_dim, thread_idx

        debug_assert(
            Int(gpuClustering.maxNumModules) < 2048,
            "fillHitsModuleStart: module count no longer fits two scans",
        )
        debug_assert(grid_dim.x == 1, "fillHitsModuleStart: requires one block")
        debug_assert(block_idx.x == 0, "fillHitsModuleStart: requires block zero")
        debug_assert(block_dim.x == 1024, "fillHitsModuleStart: requires 1024 threads")

        var first = Int(thread_idx.x)
        var i = first
        while i < Int(gpuClustering.maxNumModules):
            var nclus = cluStart[i]
            moduleStart[i + 1] = (
                gpuClustering.maxHitsInModule if nclus > gpuClustering.maxHitsInModule else nclus
            )
            i += Int(block_dim.x)

        var ws = stack_allocation[
            32, UInt32, address_space=AddressSpace.SHARED
        ]()
        blockPrefixScan(moduleStart + 1, UInt32(1024), ws)
        blockPrefixScan(
            moduleStart + 1025,
            UInt32(gpuClustering.maxNumModules) - UInt32(1024),
            ws,
        )

        i = first + 1025
        while i < Int(gpuClustering.maxNumModules) + 1:
            moduleStart[i] += moduleStart[1024]
            i += Int(block_dim.x)
        barrier()
