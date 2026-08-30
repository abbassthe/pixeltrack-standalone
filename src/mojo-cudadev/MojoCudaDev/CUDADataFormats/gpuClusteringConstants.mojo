# Mojo port of CUDADataFormats/gpuClusteringConstants.h.
from sys import is_defined


struct gpuClustering:
    @staticmethod
    fn maxHitsInIter() -> UInt32:
        if is_defined["GPU_SMALL_EVENTS"]():
            return 64
        else:
            return 160

    comptime maxHitsInModule: UInt32 = 1024
    comptime maxNumModules: UInt16 = 2000
    comptime maxNumClustersPerModules: Int32 = 1024
    # invalidModuleId must be greater than maxNumModules
    comptime invalidModuleId: UInt16 = 65534
