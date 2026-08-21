from memory import stack_allocation
from std.memory import AddressSpace

from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add, atomic_fetch_inc_wrap, atomic_fetch_min
from MojoCudaDev.CUDACore.CUDASync import syncthreads_or
from MojoCudaDev.CUDACore.HistoContainer import HistoContainer
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering as gpuClusteringConstants
from MojoCudaDev.Geometry.Phase1PixelTopology import Phase1PixelTopology


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
