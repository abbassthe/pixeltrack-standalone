# Mojo port of plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.{h,cu}.
# The producer that fills WordFedAppender from FEDRawDataCollection is added
# separately; this object owns the asynchronous device pipeline and returns
# the two CUDA products after its producing stream has completed.
from memory import memcpy
from std.gpu.host import DeviceContext
from std.gpu.host.dim import Dim
from std.sys.info import size_of

from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.CUDACore.currentDevice import currentDevice
from MojoCudaDev.CUDACore.device_unique_ptr import (
    unique_ptr as DeviceUniquePtr,
    make_device_unique,
)
from MojoCudaDev.CUDACore.host_noncached_unique_ptr import (
    unique_ptr as HostNoncachedUniquePtr,
    make_host_noncached_unique,
)
from MojoCudaDev.CUDACore.host_unique_ptr import (
    unique_ptr as HostUniquePtr,
    make_host_unique,
    _HostAllocation,
)
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA
from MojoCudaDev.CUDADataFormats.SiPixelDigiErrorsCUDA import (
    SiPixelDigiErrorsCUDA,
    SiPixelErrorCompactVector,
)
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import SiPixelDigisCUDA
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoCudaDev.CondFormats.SiPixelROCsStatusAndMapping import SiPixelROCsStatusAndMapping
from MojoCudaDev.DataFormats.SiPixelFormatterErrors import SiPixelFormatterErrors
from MojoCudaDev.MojoBridge.DTypes import Typeable, cudaHostAllocWriteCombined
from MojoCudaDev.plugin_SiPixelClusterizer.RawToDigi import RawToDigi
from MojoCudaDev.plugin_SiPixelClusterizer.SiPixelClusterThresholds import (
    SiPixelClusterThresholds,
)
from MojoCudaDev.plugin_SiPixelClusterizer.SiPixelRawToClusterKernels import (
    SiPixelRawToClusterKernels,
)
from MojoCudaDev.plugin_SiPixelClusterizer.gpuCalibPixel import gpuCalibPixel
from MojoCudaDev.plugin_SiPixelClusterizer.gpuClusterChargeCut import (
    gpuClusterChargeCut,
)
from MojoCudaDev.plugin_SiPixelClusterizer.gpuClustering import gpuClustering as ClusteringKernels


comptime MAX_FED_WORDS: UInt = 150 * 2000
comptime _RAW_THREADS = 512
comptime _CLUSTER_THREADS = 256


fn _null_host_ptr[T: AnyType]() -> HostUniquePtr[T]:
    return HostUniquePtr[T](_HostAllocation[T]())


# C++ SiPixelRawToClusterGPUKernel::WordFedAppender. Its buffers are mapped
# write-combined host allocations, so the GPU copy kernels can safely read
# them after makeClustersAsync() returns.
struct WordFedAppender(Movable):
    var word_: HostNoncachedUniquePtr[UInt32]
    var fedId_: HostNoncachedUniquePtr[UInt8]

    # Framework producer constructors cannot raise. Allocation therefore
    # happens on first event in ensureAllocated(), inside produce()'s error
    # boundary, while retaining C++'s persistent appender lifetime afterward.
    fn __init__(out self):
        self.word_ = HostNoncachedUniquePtr[UInt32](UnsafePointer[UInt32, MutAnyOrigin]())
        self.fedId_ = HostNoncachedUniquePtr[UInt8](UnsafePointer[UInt8, MutAnyOrigin]())

    fn ensureAllocated(mut self) raises:
        if self.word_.get() != UnsafePointer[UInt32, MutAnyOrigin]():
            return
        self.word_ = make_host_noncached_unique[UInt32](
            MAX_FED_WORDS, cudaHostAllocWriteCombined
        )
        self.fedId_ = make_host_noncached_unique[UInt8](
            MAX_FED_WORDS, cudaHostAllocWriteCombined
        )

    fn __moveinit__(out self, deinit take: Self):
        self.word_ = take.word_^
        self.fedId_ = take.fedId_^

    fn initializeWordFed(
        mut self,
        fedId: Int32,
        wordCounterGPU: UInt32,
        src: UnsafePointer[UInt32, MutAnyOrigin],
        length: UInt32,
    ):
        debug_assert(
            UInt(wordCounterGPU) + UInt(length) <= MAX_FED_WORDS,
            "WordFedAppender: FED words exceed MAX_FED_WORDS",
        )
        memcpy(
            dest=(self.word_.get() + Int(wordCounterGPU)).bitcast[UInt8](),
            src=src.bitcast[UInt8](),
            count=Int(length) * size_of[UInt32](),
        )
        var value = UInt8(fedId - 1200)
        var i = 0
        while i < Int(length) // 2:
            self.fedId_.get()[Int(wordCounterGPU) // 2 + i] = value
            i += 1

    fn word(self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.word_.get()

    fn fedId(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return self.fedId_.get()


struct ClusterResults(Movable):
    # Named counterpart of C++ std::pair<SiPixelDigisCUDA, SiPixelClustersCUDA>.
    # Tuple destructuring currently copies non-copyable elements in this Mojo
    # release, so expose the same two ownership transfers as named fields.
    var digis: SiPixelDigisCUDA
    var clusters: SiPixelClustersCUDA

    fn __init__(out self, var digisIn: SiPixelDigisCUDA, var clustersIn: SiPixelClustersCUDA):
        self.digis = digisIn^
        self.clusters = clustersIn^

    fn __moveinit__(out self, deinit take: Self):
        self.digis = take.digis^
        self.clusters = take.clusters^


struct SiPixelRawToClusterGPUKernel(Movable, Typeable):
    var nDigis: UInt32
    # C++ nModules_Clusters_h: [number of active modules, number of clusters].
    # It remains alive until getResults() after the producer completion event.
    var nModules_Clusters_h: HostUniquePtr[UInt32]
    var digis_d: SiPixelDigisCUDA
    var clusters_d: SiPixelClustersCUDA
    var digiErrors_d: SiPixelDigiErrorsCUDA
    # C++ keeps these as local stream-ordered device pointers. This port's
    # allocator does not provide stream-ordered release, so retain them until
    # getResults() to keep the RawToDigi launch safe.
    var word_d: DeviceUniquePtr[UInt32]
    var fedId_d: DeviceUniquePtr[UInt8]

    fn __init__(out self):
        self.nDigis = 0
        self.nModules_Clusters_h = _null_host_ptr[UInt32]()
        self.digis_d = SiPixelDigisCUDA()
        self.clusters_d = SiPixelClustersCUDA()
        self.digiErrors_d = SiPixelDigiErrorsCUDA()
        self.word_d = DeviceUniquePtr[UInt32]()
        self.fedId_d = DeviceUniquePtr[UInt8]()

    fn __moveinit__(out self, deinit take: Self):
        self.nDigis = take.nDigis
        self.nModules_Clusters_h = take.nModules_Clusters_h^
        self.digis_d = take.digis_d^
        self.clusters_d = take.clusters_d^
        self.digiErrors_d = take.digiErrors_d^
        self.word_d = take.word_d^
        self.fedId_d = take.fedId_d^

    fn makeClustersAsync(
        mut self,
        isRun2: Bool,
        clusterThresholds: SiPixelClusterThresholds,
        cablingMap: UnsafePointer[SiPixelROCsStatusAndMapping, MutAnyOrigin],
        modToUnp: UnsafePointer[UInt8, MutAnyOrigin],
        gains: UnsafePointer[SiPixelGainForHLTonGPU, MutAnyOrigin],
        wordFed: WordFedAppender,
        var errors: SiPixelFormatterErrors,
        wordCounter: UInt32,
        fedCounter: UInt32,
        useQualityInfo: Bool,
        includeErrors: Bool,
        debug: Bool,
        mut dev_state: _AllocateDeviceState,
        mut host_state: _AllocateHostState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        _ = fedCounter
        self.nDigis = wordCounter
        self.digis_d = SiPixelDigisCUDA(MAX_FED_WORDS, dev_state, host_state, stream)
        self.clusters_d = SiPixelClustersCUDA(
            UInt(gpuClustering.maxNumModules), dev_state, host_state, stream
        )
        self.nModules_Clusters_h = make_host_unique[UInt32](2, host_state, stream)
        self.word_d = DeviceUniquePtr[UInt32]()
        self.fedId_d = DeviceUniquePtr[UInt8]()
        self.digiErrors_d = SiPixelDigiErrorsCUDA()
        if includeErrors:
            self.digiErrors_d = SiPixelDigiErrorsCUDA(
                MAX_FED_WORDS, errors^, dev_state, host_state, stream
            )

        var deviceStream = stream.get()
        var ctx = DeviceContext(api="cuda", device_id=currentDevice())

        if wordCounter != 0:
            debug_assert(wordCounter % 2 == 0, "RawToCluster: word count must be even")
            self.word_d = make_device_unique[UInt32](UInt(wordCounter), dev_state, stream)
            self.fedId_d = make_device_unique[UInt8](UInt(wordCounter), dev_state, stream)
            copyAsync[UInt32](self.word_d, wordFed.word(), UInt(wordCounter), stream)
            copyAsync[UInt8](self.fedId_d, wordFed.fedId(), UInt(wordCounter) // 2, stream)

            var raw_to_digi = ctx.compile_function[RawToDigi.convert, RawToDigi.convert]()
            var raw_blocks = (Int(wordCounter) + _RAW_THREADS - 1) // _RAW_THREADS
            var error_ptr = (
                self.digiErrors_d.error()
                if includeErrors
                else UnsafePointer[SiPixelErrorCompactVector, MutAnyOrigin]()
            )
            deviceStream.enqueue_function(
                raw_to_digi,
                cablingMap,
                modToUnp,
                wordCounter,
                self.word_d.get(),
                self.fedId_d.get(),
                self.digis_d.xx(),
                self.digis_d.yy(),
                self.digis_d.adc(),
                self.digis_d.pdigi(),
                self.digis_d.rawIdArr(),
                self.digis_d.moduleInd(),
                error_ptr,
                Int32(1) if useQualityInfo else Int32(0),
                Int32(1) if includeErrors else Int32(0),
                Int32(1) if debug else Int32(0),
                grid_dim=Dim(raw_blocks),
                block_dim=Dim(_RAW_THREADS),
            )
            if includeErrors:
                self.digiErrors_d.copyErrorToHostAsync(stream)

        # C++ begins these stages even for an empty event, because calibDigis
        # zeroes the counters consumed by the later kernels.
        var work_items = (
            Int(wordCounter)
            if Int(wordCounter) > Int(gpuClustering.maxNumModules)
            else Int(gpuClustering.maxNumModules)
        )
        var cluster_blocks = (work_items + _CLUSTER_THREADS - 1) // _CLUSTER_THREADS

        var calib = ctx.compile_function[gpuCalibPixel.calibDigis, gpuCalibPixel.calibDigis]()
        deviceStream.enqueue_function(
            calib,
            Int32(1) if isRun2 else Int32(0),
            self.digis_d.moduleInd(),
            self.digis_d.xx(),
            self.digis_d.yy(),
            self.digis_d.adc(),
            gains,
            Int32(wordCounter),
            self.clusters_d.moduleStart(),
            self.clusters_d.clusInModule(),
            self.clusters_d.clusModuleStart(),
            grid_dim=Dim(cluster_blocks),
            block_dim=Dim(_CLUSTER_THREADS),
        )

        var count_modules = ctx.compile_function[
            ClusteringKernels.countModules, ClusteringKernels.countModules
        ]()
        deviceStream.enqueue_function(
            count_modules,
            self.digis_d.moduleInd(),
            self.clusters_d.moduleStart(),
            self.digis_d.clus(),
            Int32(wordCounter),
            grid_dim=Dim(cluster_blocks),
            block_dim=Dim(_CLUSTER_THREADS),
        )
        copyAsync[UInt32](
            self.nModules_Clusters_h,
            self.clusters_d.moduleStart(),
            UInt(1),
            stream,
        )

        var find_clus = ctx.compile_function[ClusteringKernels.findClus, ClusteringKernels.findClus]()
        deviceStream.enqueue_function(
            find_clus,
            self.digis_d.moduleInd(),
            self.digis_d.xx(),
            self.digis_d.yy(),
            self.clusters_d.moduleStart(),
            self.clusters_d.clusInModule(),
            self.clusters_d.moduleId(),
            self.digis_d.clus(),
            Int32(wordCounter),
            UnsafePointer[UInt32, MutAnyOrigin](),
            grid_dim=Dim(Int(gpuClustering.maxNumModules)),
            block_dim=Dim(_CLUSTER_THREADS),
        )

        var charge_cut = ctx.compile_function[
            gpuClusterChargeCut.clusterChargeCut,
            gpuClusterChargeCut.clusterChargeCut,
        ]()
        deviceStream.enqueue_function(
            charge_cut,
            clusterThresholds,
            self.digis_d.moduleInd(),
            self.digis_d.adc(),
            self.clusters_d.moduleStart(),
            self.clusters_d.clusInModule(),
            self.clusters_d.moduleId(),
            self.digis_d.clus(),
            wordCounter,
            grid_dim=Dim(Int(gpuClustering.maxNumModules)),
            block_dim=Dim(_CLUSTER_THREADS),
        )

        var fill_starts = ctx.compile_function[
            SiPixelRawToClusterKernels.fillHitsModuleStart,
            SiPixelRawToClusterKernels.fillHitsModuleStart,
        ]()
        deviceStream.enqueue_function(
            fill_starts,
            self.clusters_d.clusInModule(),
            self.clusters_d.clusModuleStart(),
            grid_dim=Dim(1),
            block_dim=Dim(1024),
        )
        copyAsync[UInt32](
            self.nModules_Clusters_h[].get() + 1,
            self.clusters_d.clusModuleStart() + Int(gpuClustering.maxNumModules),
            UInt(1),
            stream,
        )

    # Call after the producing ScopedContext's completion event, exactly where
    # C++ invokes getResults() from SiPixelRawToClusterCUDA::produce().
    fn getResults(mut self) -> ClusterResults:
        self.digis_d.setNModulesDigis(self.nModules_Clusters_h[].get()[0], self.nDigis)
        self.clusters_d.setNClusters(self.nModules_Clusters_h[].get()[1])
        var digis = self.digis_d^
        var clusters = self.clusters_d^
        self.digis_d = SiPixelDigisCUDA()
        self.clusters_d = SiPixelClustersCUDA()
        self.word_d = DeviceUniquePtr[UInt32]()
        self.fedId_d = DeviceUniquePtr[UInt8]()
        self.nModules_Clusters_h = _null_host_ptr[UInt32]()
        return ClusterResults(digis^, clusters^)

    fn getErrors(mut self) -> SiPixelDigiErrorsCUDA:
        var errors = self.digiErrors_d^
        self.digiErrors_d = SiPixelDigiErrorsCUDA()
        return errors^

    @staticmethod
    fn dtype() -> String:
        return "SiPixelRawToClusterGPUKernel"
