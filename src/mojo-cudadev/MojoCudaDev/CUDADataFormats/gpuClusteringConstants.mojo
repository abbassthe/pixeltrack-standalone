# Mojo port of CUDADataFormats/gpuClusteringConstants.h.
comptime GPU_SMALL_EVENTS = False  # matches TrackSoAHeterogeneousT.mojo's own #ifdef modeling


struct gpuClustering:
    @staticmethod
    fn maxHitsInIter() -> UInt32:
        if GPU_SMALL_EVENTS:
            return 64
        else:
            return 160

    comptime maxHitsInModule: UInt32 = 1024
    comptime maxNumModules: UInt16 = 2000
    comptime maxNumClustersPerModules: Int32 = 1024
    # invalidModuleId must be greater than maxNumModules
    comptime invalidModuleId: UInt16 = 65534
