# Mojo port of plugin-SiPixelClusterizer/gpuCalibPixel.h. Pure rename sweep
# from cuda (confirmed via diff): InvId became gpuClustering::invalidModuleId,
# MaxNumModules became maxNumModules -- no logic changes.
#
# Unlike mojo-serial's GPUCalibPixel.mojo (a simplified single-threaded CPU
# port -- plain sequential for-loop, no grid striding, nClustersInModule
# zeroed via memset), this is a real GPU kernel: grid-stride loops over
# blockDim.x*blockIdx.x+threadIdx.x, matching cudadev's __global__ calibDigis
# exactly, following the grid-stride idiom established in
# OneToManyAssoc.mojo's bulkFinalizeFill. C++'s `if (...) continue;` becomes
# an if-wrapped loop body instead (a while-loop continue would skip the
# stride increment).
#
# isRun2 is Int32 (0/nonzero), not Bool: calibDigis is meant to be launched
# directly as a kernel (matching its C++ __global__ role, confirmed by a
# minimal spike), and Mojo's enqueue_function requires every argument to
# conform to DevicePassable -- Bool doesn't (confirmed the same way), while
# Int32 already does (established by memsetAsync.mojo's value/nbytes args).
from MojoCudaDev.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.MojoBridge.DTypes import Float


struct gpuCalibPixel:
    # valid for run2
    comptime VCaltoElectronGain: Float = 47  # L2-4: 47 +- 4.7
    comptime VCaltoElectronGain_L1: Float = 50  # L1:   49.6 +- 2.6
    comptime VCaltoElectronOffset: Float = -60  # L2-4: -60 +- 130
    comptime VCaltoElectronOffset_L1: Float = -670  # L1:   -670 +- 220

    @staticmethod
    fn calibDigis(
        isRun2: Int32,
        id: UnsafePointer[UInt16, MutAnyOrigin],
        x: UnsafePointer[UInt16, MutAnyOrigin],
        y: UnsafePointer[UInt16, MutAnyOrigin],
        adc: UnsafePointer[UInt16, MutAnyOrigin],
        ped: UnsafePointer[SiPixelGainForHLTonGPU, MutAnyOrigin],
        numElements: Int32,
        moduleStart: UnsafePointer[UInt32, MutAnyOrigin],  # just to zero first
        nClustersInModule: UnsafePointer[UInt32, MutAnyOrigin],  # just to zero them
        clusModuleStart: UnsafePointer[UInt32, MutAnyOrigin],  # just to zero first
    ):
        from std.gpu import thread_idx, block_idx, block_dim, grid_dim

        var first = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        var stride = Int(grid_dim.x) * Int(block_dim.x)
        var isRun2Bool = isRun2 != 0

        # zero for next kernels...
        if first == 0:
            clusModuleStart[0] = 0
            moduleStart[0] = 0
        var i = first
        while i < Int(gpuClustering.maxNumModules):
            nClustersInModule[i] = 0
            i += stride

        i = first
        while i < Int(numElements):
            if gpuClustering.invalidModuleId != id[i]:
                var conversionFactor: Float = (
                    Self.VCaltoElectronGain_L1 if id[i] < 96 else Self.VCaltoElectronGain
                ) if isRun2Bool else Float(1.0)
                var offset: Float = (
                    Self.VCaltoElectronOffset_L1 if id[i] < 96 else Self.VCaltoElectronOffset
                ) if isRun2Bool else Float(0.0)

                var isDeadColumn = False
                var isNoisyColumn = False

                var row = x[i].cast[DType.int32]()
                var col = y[i].cast[DType.int32]()
                var ret = ped[].getPedAndGain(
                    id[i].cast[DType.uint32](), col, row, isDeadColumn, isNoisyColumn
                )
                var pedestal = ret[0]
                var gain = ret[1]
                if isDeadColumn or isNoisyColumn:
                    id[i] = gpuClustering.invalidModuleId
                    adc[i] = 0
                    print("bad pixel at", i, "in", id[i])
                else:
                    var rawAdc = adc[i].cast[DType.float32]()

                    # Match C++ float arithmetic step-by-step:
                    # float vcal = adc[i] * gain - pedestal * gain;
                    var vcal = (rawAdc * gain - pedestal * gain).cast[DType.float32]()

                    # Match C++: adc[i] = std::max(100, int(vcal * conversionFactor + offset));
                    var calibratedF = (vcal * conversionFactor + offset).cast[DType.float32]()
                    var calibrated = calibratedF.cast[DType.int32]()  # truncates toward zero, matching C++'s int()

                    adc[i] = UInt16(100) if calibrated < 100 else calibrated.cast[DType.uint16]()
            i += stride
